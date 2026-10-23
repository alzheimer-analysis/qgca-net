"""视频预处理：解析 OpenFace 2.0 输出、人脸对齐 + ViT-Base 嵌入、视频保真度指标。

每帧特征 22 维 = 17 个 AU 强度 + 3 个头部姿态角 + 2 个视线分量。
帧检测失败标记为无效（不插值）。话语内有效帧均值池化在数据集读取时完成（便于增强）。

视频保真度（越大越好）:
  tracking_confidence  OpenFace 地标跟踪置信度（话语内所有帧的平均，失败帧计 0）
  sharpness            人脸裁剪的拉普拉斯方差
  illumination         1 - |人脸平均亮度 - 0.5| / 0.5（与中灰偏差，取反）
DAIC-WOZ 不发布原始视频：用其发布的 HOG(4464) + 3D 地标(204) 描述子代替 ViT 嵌入，且只有跟踪置信度可用。
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AU_COLS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
           "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
POSE_COLS = ["pose_Rx", "pose_Ry", "pose_Rz"]
GAZE_COLS = ["gaze_angle_x", "gaze_angle_y"]
APP_FPS = 2.0                      # ViT 嵌入与 DAIC-WOZ 外观描述子的采样率（帧/秒）


# ---------------------------------------------------------------------------- OpenFace 解析
def read_openface_csv(path: str) -> Dict[str, np.ndarray]:
    """OpenFace 2.0 FeatureExtraction 的 CSV（-aus -pose -gaze -2Dfp）。"""
    import pandas as pd
    df = pd.read_csv(path, skipinitialspace=True)
    feats = np.stack([df[c].to_numpy(np.float32) if c in df else np.zeros(len(df), np.float32)
                      for c in AU_COLS + POSE_COLS + GAZE_COLS], 1)
    lm = None
    xs, ys = [f"x_{i}" for i in range(68)], [f"y_{i}" for i in range(68)]
    if all(c in df for c in xs + ys):
        lm = np.stack([df[xs].to_numpy(np.float32), df[ys].to_numpy(np.float32)], -1)
    return {"t": df["timestamp"].to_numpy(np.float64), "conf": df["confidence"].to_numpy(np.float32),
            "ok": df["success"].to_numpy().astype(bool), "feats": feats, "lm2d": lm}


def read_daic_visual(session_dir: str, pid: str, hog_dim: int = 4464) -> Dict[str, np.ndarray]:
    """DAIC-WOZ 发布的逐帧视觉特征: _CLNF_AUs / _CLNF_pose / _CLNF_gaze / _CLNF_hog / _CLNF_features3D。

    DAIC-WOZ 只提供部分 AU 强度，缺失的 AU 列以 0 补齐到 17 维。外观描述子 = [HOG ; 3D 地标]，按 2 帧/秒采样。
    """
    import os
    import pandas as pd
    rd = lambda suffix: pd.read_csv(os.path.join(session_dir, f"{pid}_CLNF_{suffix}.txt"), skipinitialspace=True)
    au, pose, gaze = rd("AUs"), rd("pose"), rd("gaze")
    feats = np.zeros((len(au), 22), np.float32)
    for j, c in enumerate(AU_COLS):
        if c in au:
            feats[:, j] = au[c].to_numpy(np.float32)
    feats[:, 17:20] = pose[["Rx", "Ry", "Rz"]].to_numpy(np.float32)
    g = gaze[["x0", "y0", "z0", "x1", "y1", "z1"]].to_numpy(np.float32)
    gx, gy, gz = (g[:, 0] + g[:, 3]) / 2, (g[:, 1] + g[:, 4]) / 2, (g[:, 2] + g[:, 5]) / 2
    feats[:, 20], feats[:, 21] = np.arctan2(gx, -gz), np.arctan2(gy, -gz)         # 等价于 OpenFace 的 gaze_angle
    t = au["timestamp"].to_numpy(np.float64)
    step = max(1, int(round(1.0 / APP_FPS / max(np.median(np.diff(t)), 1e-3))))
    hog = rd("hog").iloc[::step]
    f3d = rd("features3D").iloc[::step]
    n = min(len(hog), len(f3d))
    app = np.concatenate([hog.iloc[:n, -hog_dim:].to_numpy(np.float32), f3d.iloc[:n, 4:208].to_numpy(np.float32)], 1)
    return {"t": t, "conf": au["confidence"].to_numpy(np.float32), "ok": au["success"].to_numpy().astype(bool),
            "feats": feats, "lm2d": None, "app": app.astype(np.float16), "app_t": t[::step][:n],
            "app_ok": au["success"].to_numpy().astype(bool)[::step][:n], "app_conf": au["confidence"].to_numpy(np.float32)[::step][:n]}


# ---------------------------------------------------------------------------- 人脸对齐、ViT、画质
def align_face(frame_bgr: np.ndarray, lm2d: np.ndarray, out_size: int = 224, margin: float = 0.25) -> np.ndarray:
    """按双眼中心连线旋转对齐，并按地标外接框（加边距）裁剪、缩放。输入 lm2d: [68,2] 像素坐标。"""
    import cv2
    right_eye, left_eye = lm2d[36:42].mean(0), lm2d[42:48].mean(0)
    angle = np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    center = tuple(((right_eye + left_eye) / 2).tolist())
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(frame_bgr, rot, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR)
    pts = np.hstack([lm2d, np.ones((68, 1))]) @ rot.T
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - margin * w)), int(min(img.shape[1], x1 + margin * w))
    y0, y1 = int(max(0, y0 - margin * h)), int(min(img.shape[0], y1 + margin * h))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size)) if crop.size else np.zeros((out_size, out_size, 3), np.uint8)


def face_quality(crop_bgr: np.ndarray) -> Tuple[float, float]:
    """(拉普拉斯方差, 光照得分)。亮度先归一到 [0,1]。"""
    import cv2
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lum = float(gray.mean() / 255.0)
    return sharp, 1.0 - abs(lum - 0.5) / 0.5


class ViTExtractor:
    """ViT-Base（google/vit-base-patch16-224）的 [CLS] 嵌入 (768)。"""

    def __init__(self, name: str = "google/vit-base-patch16-224-in21k", device: str = "cuda"):
        import torch
        from transformers import ViTImageProcessor, ViTModel
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.proc = ViTImageProcessor.from_pretrained(name)
        self.model = ViTModel.from_pretrained(name).to(self.device).eval()

    def __call__(self, crops_bgr: Sequence[np.ndarray], batch: int = 32) -> np.ndarray:
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(crops_bgr), batch):
                rgb = [c[:, :, ::-1] for c in crops_bgr[i:i + batch]]
                x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
                outs.append(self.model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float16) if outs else np.zeros((0, 768), np.float16)


def sample_video_frames(video_path: str, of: Dict, fps: float = APP_FPS, vit: Optional[ViTExtractor] = None,
                        corrupt_ratio: float = 0.0, rng: Optional[np.random.Generator] = None) -> Dict:
    """以 fps 抽帧：对齐人脸 -> ViT 嵌入，并记录清晰度与光照。corrupt_ratio>0 时随机遮挡该比例的帧（鲁棒性扰动）。"""
    import cv2
    rng = rng or np.random.default_rng(0)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))
    ts, crops, sharp, illum, ok = [], [], [], [], []
    for fi in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        good, frame = cap.read()
        if not good:
            break
        t = fi / src_fps
        j = int(np.argmin(np.abs(of["t"] - t)))
        valid = bool(of["ok"][j]) and of["lm2d"] is not None
        if valid and corrupt_ratio > 0 and rng.random() < corrupt_ratio:
            h, w = frame.shape[:2]
            frame[:] = 0                                          # 整帧遮挡
            valid = False
        if valid:
            crop = align_face(frame, of["lm2d"][j])
            s, il = face_quality(crop)
            crops.append(crop)
        else:
            s, il = np.nan, np.nan
        ts.append(t); sharp.append(s); illum.append(il); ok.append(valid)
    cap.release()
    emb = np.zeros((len(ts), 768), np.float16)
    if vit is not None and crops:
        emb[np.flatnonzero(ok)] = vit(crops)
    return {"t": np.asarray(ts), "app": emb, "ok": np.asarray(ok, bool), "sharp": np.asarray(sharp),
            "illum": np.asarray(illum)}


# ---------------------------------------------------------------------------- 按话语切分
def utterance_video_fields(segments: Sequence[Tuple[float, float]], of: Dict, app: Optional[Dict],
                           occlusion_ratio: float = 0.0, rng: Optional[np.random.Generator] = None) -> Dict:
    """把帧级数据按话语区间切开。occlusion_ratio>0 时随机把该比例的 OpenFace 帧置为无效并将其置信度置 0。"""
    rng = rng or np.random.default_rng(0)
    ok_all, conf_all = of["ok"].copy(), of["conf"].copy()
    if occlusion_ratio > 0:
        occ = rng.random(len(ok_all)) < occlusion_ratio
        ok_all &= ~occ
        conf_all = np.where(occ, 0.0, conf_all)
    of_frames, of_valid, of_conf, app_frames, app_valid = [], [], [], [], []
    fid = np.full((len(segments), 3), np.nan)
    behav_pose, behav_gaze = np.full(len(segments), np.nan), np.full(len(segments), np.nan)
    for i, (s, e) in enumerate(segments):
        m = (of["t"] >= s) & (of["t"] < e)
        of_frames.append(of["feats"][m])
        of_valid.append(ok_all[m])
        of_conf.append(conf_all[m])
        if m.any():
            fid[i, 0] = float(conf_all[m].mean())                     # 失败/遮挡帧计 0
            good = of["feats"][m][ok_all[m]]
            if len(good):
                behav_pose[i] = float(np.abs(good[:, 17:20] - good[:, 17:20].mean(0)).mean())   # 头部运动幅度
                behav_gaze[i] = float(np.abs(good[:, 20:22] - good[:, 20:22].mean(0)).mean())   # 视线离散度
        if app is not None:
            am = (app["t"] >= s) & (app["t"] < e)
            app_frames.append(app["app"][am])
            av = app["ok"][am] if "ok" in app else np.ones(am.sum(), bool)
            if occlusion_ratio > 0:                                    # 与 OpenFace 帧同步遮挡
                av = av & (rng.random(len(av)) >= occlusion_ratio)
            app_valid.append(av)
            if am.any():
                if "sharp" in app and np.isfinite(app["sharp"][am]).any():
                    fid[i, 1] = float(np.nanmean(app["sharp"][am]))
                    fid[i, 2] = float(np.nanmean(app["illum"][am]))
        else:
            app_frames.append(np.zeros((0, 1), np.float16))
            app_valid.append(np.zeros(0, bool))
    return {"of_frames": of_frames, "of_valid": of_valid, "of_conf": of_conf, "app_frames": app_frames,
            "app_valid": app_valid, "fid": fid, "head_pose": behav_pose, "gaze": behav_gaze}
