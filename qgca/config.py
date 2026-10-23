"""YAML 配置加载：支持 `_base_` 继承与 `a.b.c=value` 形式的命令行覆盖。"""
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class Cfg(dict):
    """可用属性访问的嵌套字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)


def _to_plain(x):
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load_raw(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        deep_update(merged, _load_raw((path.parent / b).resolve()))
    return deep_update(merged, raw)


def set_by_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(d: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    for item in overrides or []:
        key, _, val = item.partition("=")
        set_by_dotted(d, key.strip(), yaml.safe_load(val))
    return d


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Cfg:
    raw = _load_raw(Path(path).resolve())
    apply_overrides(raw, overrides)
    return _wrap(raw)


def update_cfg(cfg: Cfg, patch: Dict[str, Any]) -> Cfg:
    """返回应用了 dotted-key 补丁的新配置（不改动原配置）。"""
    raw = copy.deepcopy(cfg.to_dict())
    for k, v in patch.items():
        set_by_dotted(raw, k, v)
    return _wrap(raw)
