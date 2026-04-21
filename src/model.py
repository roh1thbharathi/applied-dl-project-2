"""
Codec-Robust Deepfake Detector — AASIST + DANN
===============================================
Baseline : AASIST (Graph Attention + Sinc filters)
On top   :
  1. Gradient Reversal Layer  -> codec-invariant encoder
  2. Temporal Attention Gate  -> focus on codec-robust regions
  3. Contrastive Loss         -> same utterance, different codec = close embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Gradient Reversal Layer
# ============================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def set_lambda(self, lam):
        self.lam = lam

    def forward(self, x):
        return GradReverse.apply(x, self.lam)


# ============================================================
# 2. AASIST Building Blocks
# ============================================================

class SincConv(nn.Module):
    """Learnable sinc bandpass filters on raw waveform (from AASIST/SincNet)."""
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        assert kernel_size % 2 == 1
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate

        # Init filterbank on mel scale
        low_hz  = 30.0
        high_hz = sample_rate / 2 - 50
        mel_lo  = 2595 * math.log10(1 + low_hz  / 700)
        mel_hi  = 2595 * math.log10(1 + high_hz / 700)
        mel_pts = torch.linspace(mel_lo, mel_hi, out_channels + 2)
        hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)

        self.low_hz_  = nn.Parameter(hz_pts[:-2].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_pts[1:-1] - hz_pts[:-2]).unsqueeze(1))

        n = (kernel_size - 1) / 2.0
        self.register_buffer('n_', 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)
        self.register_buffer('window_', torch.hamming_window(kernel_size)[:kernel_size // 2])

    def forward(self, x):
        low  = torch.clamp(self.low_hz_, min=50.0)
        high = low + torch.abs(self.band_hz_)
        high = torch.clamp(high, max=self.sample_rate / 2 - 50)

        f_lo = (2 * low  / self.sample_rate) * self.n_
        f_hi = (2 * high / self.sample_rate) * self.n_

        sinc_lo = torch.sin(f_lo) / (self.n_ + 1e-9)
        sinc_hi = torch.sin(f_hi) / (self.n_ + 1e-9)

        band = (2 * high * sinc_hi - 2 * low * sinc_lo) * self.window_
        center = (2 * high - 2 * low)
        kernel = torch.cat([band, center, torch.flip(band, [-1])], dim=-1)
        kernel = kernel / (2 * kernel.norm(dim=-1, keepdim=True) + 1e-9)

        return F.conv1d(x, kernel.unsqueeze(1), padding=self.kernel_size // 2)


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer (simplified for audio feature graphs)."""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky   = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (B, N, D)  — N nodes (frequency bins or time frames)
        h   = self.W(x)                         # (B, N, out_dim)
        B, N, D = h.shape
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e   = self.leaky(self.att(torch.cat([h_i, h_j], dim=-1))).squeeze(-1)  # (B, N, N)
        a   = torch.softmax(e, dim=-1)
        a   = self.dropout(a)
        out = torch.bmm(a, h)                   # (B, N, out_dim)
        return F.elu(out)


class AASISTEncoder(nn.Module):
    """
    Simplified AASIST-style encoder.
    Raw waveform -> SincConv -> ResBlocks -> Graph Attention
                 -> Temporal Attention Gate -> global pool -> embedding

    The Temporal Attention Gate is applied BEFORE pooling, operating on the
    32-node sequence produced by GAT.  This is meaningful: each node is a
    temporal region, and attention suppresses codec-noise regions while
    amplifying codec-robust regions (pitch, prosody).  Placing it after
    pooling (on a single token) would make self-attention a trivial identity.
    """
    def __init__(self, embed_dim=256, sinc_ch=70, sample_rate=16000, n_attn_heads=4, use_temporal_attn=True):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn
        self.sinc    = SincConv(sinc_ch, kernel_size=1024 + 1, sample_rate=sample_rate)
        self.bn_sinc = nn.BatchNorm1d(sinc_ch)

        # Residual conv stack
        def res_block(in_ch, out_ch, dilation=1):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_ch), nn.GELU(),
                nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_ch),
            )

        self.res1      = res_block(sinc_ch, 128, 1)
        self.res2      = res_block(128, 128, 2)
        self.res3      = res_block(128, 128, 4)
        self.proj_skip = nn.Conv1d(sinc_ch, 128, 1)
        self.pool      = nn.AdaptiveAvgPool1d(32)

        # Graph attention on the 32 pooled frames
        self.gat1 = GraphAttentionLayer(128, 128)
        self.gat2 = GraphAttentionLayer(128, 128)

        # ── NEW: Temporal Attention Gate on the 32-node sequence ──────────
        # Operates on (B, 32, 128) — 32 temporal nodes, each 128-dim.
        # Multi-head self-attention lets each node ask: "which other regions
        # share codec-robust patterns with me?" and amplify those signals.
        self.temporal_attn = TemporalAttention(128, n_heads=n_attn_heads)

        self.head = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, waveform):
        # waveform: (B, T)
        x    = waveform.unsqueeze(1)       # (B, 1, T)
        x    = torch.abs(self.sinc(x))
        x    = F.gelu(self.bn_sinc(x))

        skip = self.proj_skip(x)
        r    = F.gelu(self.res1(x) + skip)
        r    = F.gelu(self.res2(r) + r)
        r    = F.gelu(self.res3(r) + r)
        r    = self.pool(r)                # (B, 128, 32)

        r    = r.permute(0, 2, 1)         # (B, 32, 128) — nodes
        r    = self.gat1(r)               # graph attention pass 1
        r    = self.gat2(r)               # graph attention pass 2

        # ── Temporal Attention Gate (new addition) ────────────────────────
        if self.use_temporal_attn:
            r = self.temporal_attn(r)      # (B, 32, 128) — codec-robust focus

        r    = r.mean(dim=1)              # (B, 128) — global mean pool over nodes

        return self.head(r)               # (B, embed_dim)


# ============================================================
# 3. Temporal Attention Gate (improvement #1 on top of AASIST)
# ============================================================

class TemporalAttention(nn.Module):
    """Transformer block; applied to the embedding before deepfake head."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 1, D) — single-token; works well when you stack windowed segments
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + a)
        x    = self.norm2(x + self.ff(x))
        return x


# ============================================================
# 4. Full Model
# ============================================================

class CodecRobustDetector(nn.Module):
    """
    AASIST encoder + DANN-style codec adversarial training.

    Forward returns dict:
      deepfake_logits : (B, 2)
      codec_logits    : (B, n_codec_classes)
      embedding       : (B, embed_dim)
    """
    def __init__(self, embed_dim=256, n_codec_classes=10, sample_rate=16000, n_attn_heads=4, use_temporal_attn=True):
        super().__init__()
        self.encoder = AASISTEncoder(embed_dim=embed_dim, sample_rate=sample_rate,
                                     n_attn_heads=n_attn_heads, use_temporal_attn=use_temporal_attn)
        self.grl     = GradientReversalLayer(lam=1.0)

        # Head 1: deepfake classifier
        self.deepfake_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        # Head 2: codec domain classifier (sits behind GRL)
        self.codec_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_codec_classes),
        )

    def set_lambda(self, lam):
        self.grl.set_lambda(lam)

    def forward(self, waveform):
        emb = self.encoder(waveform)               # (B, D)
        # Temporal attention is applied inside encoder on 32-node sequence

        return {
            'deepfake_logits': self.deepfake_head(emb),
            'codec_logits':    self.codec_head(self.grl(emb)),
            'embedding':       emb,
        }


# ============================================================
# 5. Contrastive Loss (improvement #2)
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE loss. Input: (2B, D) where rows 2i and 2i+1 are a positive pair
    (same utterance, different codec).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        emb = F.normalize(embeddings, dim=-1)
        B2  = emb.size(0)
        assert B2 % 2 == 0

        sim     = torch.mm(emb, emb.T) / self.temperature
        mask    = torch.eye(B2, device=emb.device).bool()
        sim     = sim.masked_fill(mask, -1e9)

        pos_idx             = torch.arange(B2, device=emb.device)
        pos_idx[0::2]      += 1
        pos_idx[1::2]      -= 1

        return F.cross_entropy(sim, pos_idx)


# ============================================================
# Quick check
# ============================================================
if __name__ == '__main__':
    model = CodecRobustDetector(embed_dim=256, n_codec_classes=10)
    x     = torch.randn(4, 16000)   # 4 clips × 1 sec @ 16 kHz
    model.set_lambda(0.5)
    out   = model(x)
    print("deepfake_logits:", out['deepfake_logits'].shape)  # (4, 2)
    print("codec_logits   :", out['codec_logits'].shape)     # (4, 10)
    print("embedding      :", out['embedding'].shape)        # (4, 256)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")