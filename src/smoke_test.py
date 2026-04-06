"""
Smoke test — no dataset needed. Run this first to verify the setup.

  cd Applied-DL-2
  python src/smoke_test.py
"""

import sys, torch, numpy as np
sys.path.insert(0, "src")

print("=" * 55)
print("  Smoke Test — Codec-Robust Deepfake Detector")
print("=" * 55)

# 1. Imports
print("\n[1/4] Imports...")
from model import CodecRobustDetector, ContrastiveLoss
from data_utils import CODEC_TO_IDX, N_CODEC_CLASSES, sample_random_codec
from evaluate import compute_eer, compute_auc
print(f"  OK  — {N_CODEC_CLASSES} codec classes")

# 2. Codec table
print(f"\n[2/4] Codec classes:")
for (c, b), i in CODEC_TO_IDX.items():
    print(f"  [{i:2d}] {c:12s} {str(b):5s} kbps")

# 3. Forward pass
print("\n[3/4] Forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CodecRobustDetector(embed_dim=128, n_codec_classes=N_CODEC_CLASSES).to(device)
x      = torch.randn(4, 16000, device=device)
model.set_lambda(0.5)
with torch.no_grad():
    out = model(x)

assert out["deepfake_logits"].shape == (4, 2)
assert out["codec_logits"].shape    == (4, N_CODEC_CLASSES)
assert out["embedding"].shape       == (4, 128)
print(f"  deepfake_logits : {tuple(out['deepfake_logits'].shape)}")
print(f"  codec_logits    : {tuple(out['codec_logits'].shape)}")
print(f"  embedding       : {tuple(out['embedding'].shape)}")

# 4. Losses + metrics
print("\n[4/4] Losses & metrics...")
import torch.nn as nn
l_df    = nn.CrossEntropyLoss()(out["deepfake_logits"], torch.randint(0,2,(4,),device=device))
l_codec = nn.CrossEntropyLoss()(out["codec_logits"],    torch.randint(0,N_CODEC_CLASSES,(4,),device=device))
pairs   = torch.stack([out["embedding"], out["embedding"]], dim=1).view(8, -1)
l_con   = ContrastiveLoss()(pairs)
print(f"  L_deepfake    = {l_df.item():.4f}")
print(f"  L_codec       = {l_codec.item():.4f}")
print(f"  L_contrastive = {l_con.item():.4f}")

scores = np.random.rand(500)
labels = np.random.randint(0, 2, 500)
print(f"  EER (random)  = {compute_eer(labels,scores)*100:.1f}%  (expect ~50%)")
print(f"  AUC (random)  = {compute_auc(labels,scores):.3f}  (expect ~0.5)")

n = sum(p.numel() for p in model.parameters())
print(f"\n{'='*55}")
print(f"  ALL CHECKS PASSED")
print(f"  Params  : {n:,}")
print(f"  Device  : {device}")
print(f"{'='*55}")