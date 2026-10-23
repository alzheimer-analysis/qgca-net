"""带"时间间隔感知相对位置编码"的 Transformer 编码器（文本/视频时序编码器使用）。

相对位置偏置：对话语起始时间差 Δ = t_j - t_i（秒），取符号分桶 sign(Δ)·floor(log2(1+|Δ|))，
每个头一个可学习标量加到注意力分数上。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


class TimeGapBias(nn.Module):
    def __init__(self, n_heads: int, n_buckets: int = 16):
        super().__init__()
        self.nb = n_buckets
        self.emb = nn.Embedding(2 * n_buckets + 1, n_heads)
        nn.init.zeros_(self.emb.weight)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """time: [B,n] 秒 -> [B,H,n,n]（query i, key j）"""
        gap = time[:, None, :] - time[:, :, None]
        b = torch.sign(gap) * torch.floor(torch.log2(1.0 + gap.abs()))
        b = b.clamp(-self.nb, self.nb).long() + self.nb
        return self.emb(b).permute(0, 3, 1, 2)


class BiasedSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = dropout

    def forward(self, x, attn_bias):
        B, n, d = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype),
                                           dropout_p=self.drop if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, n, d))


class BiasedEncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = BiasedSelfAttention(d, heads, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        x = self.ln1(x + self.drop(self.attn(x, attn_bias)))
        return self.ln2(x + self.drop(self.ffn(x)))


class TimeGapTransformer(nn.Module):
    """x:[B,n,d], time:[B,n], seq_mask:[B,n] -> [B,n,d]"""

    def __init__(self, d: int, heads: int, layers: int, ffn_mult: int = 4, dropout: float = 0.1,
                 time_aware: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([BiasedEncoderLayer(d, heads, ffn_mult, dropout) for _ in range(layers)])
        self.bias = nn.ModuleList([TimeGapBias(heads) for _ in range(layers)]) if time_aware else None
        self.heads = heads

    def forward(self, x, time, seq_mask):
        pad = (1.0 - seq_mask)[:, None, None, :] * NEG                 # 屏蔽补零的键
        for i, layer in enumerate(self.layers):
            bias = pad + (self.bias[i](time) if self.bias is not None else 0.0)
            bias = bias.expand(x.size(0), self.heads, x.size(1), x.size(1))
            x = layer(x, bias)
        return x * seq_mask.unsqueeze(-1)
