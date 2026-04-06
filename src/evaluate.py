"""
Evaluation: EER, AUC, per-codec breakdown, spectral viz, t-SNE.

Usage:
  python src/evaluate.py \
    --checkpoint results/best_model.pt \
    --asv_root   /path/to/ASVspoof2019 \
    --output_dir results/ \
    --tsne
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    """EER: point where FAR == FRR. labels: 0=real 1=fake, scores: P(fake)."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr  = 1 - tpr
    idx  = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(sk_auc(fpr, tpr))


# ── Per-codec breakdown ───────────────────────────────────────────────────────

@torch.no_grad()
def per_codec_eval(model, asv_root, device, split="eval", batch_size=32, num_workers=4):
    from data_utils import CODEC_CONFIG, ASVspoof2019Dataset
    from torch.utils.data import DataLoader

    rows = []
    for codec, bitrates in CODEC_CONFIG.items():
        for bitrate in bitrates:
            ds = ASVspoof2019Dataset(asv_root, split=split, random_codec=False,
                                     codec=codec, bitrate=bitrate, contrastive=False)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
            sc, lb = [], []
            model.eval()
            for batch in loader:
                out = model(batch["waveform"].to(device))
                sc.append(torch.softmax(out["deepfake_logits"], -1)[:, 1].cpu().numpy())
                lb.append(batch["label"].numpy())
            scores = np.concatenate(sc)
            labels = np.concatenate(lb)
            eer = compute_eer(labels, scores)
            auc = compute_auc(labels, scores)
            label_str = f"{codec}/{bitrate}kbps" if bitrate else codec
            rows.append({"codec": codec, "bitrate": bitrate or "—",
                         "EER%": round(eer * 100, 2), "AUC": round(auc, 4), "n": len(ds)})
            print(f"  {label_str:20s} | EER={eer*100:.2f}%  AUC={auc:.4f}")
    return pd.DataFrame(rows)


# ── t-SNE plot ────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_tsne(model, loader, device, save_path="results/tsne.png", max_n=2000):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    embs, lbls, cdcs = [], [], []
    for batch in loader:
        if sum(len(e) for e in embs) >= max_n:
            break
        out = model(batch["waveform"].to(device))
        embs.append(out["embedding"].cpu().numpy())
        lbls.append(batch["label"].numpy())
        cdcs.append(batch["codec_idx"].numpy())

    emb = np.concatenate(embs)[:max_n]
    lbl = np.concatenate(lbls)[:max_n]
    cdc = np.concatenate(cdcs)[:max_n]

    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for v, name, color in [(0, "Real", "#2196F3"), (1, "Fake", "#F44336")]:
        m = lbl == v
        ax1.scatter(proj[m, 0], proj[m, 1], c=color, label=name, alpha=0.5, s=8)
    ax1.set_title("Real vs Fake"); ax1.legend(); ax1.axis("off")

    sc = ax2.scatter(proj[:, 0], proj[:, 1], c=cdc, cmap="tab10", alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax2, label="Codec class")
    ax2.set_title("Codec type (should be mixed for good invariance)"); ax2.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"t-SNE saved → {save_path}")


# ── Spectral diff plot ────────────────────────────────────────────────────────

def plot_spectral_diff(audio_path, save_path="results/spectral_diff.png", sr=16000):
    import torchaudio, torchaudio.transforms as T, matplotlib.pyplot as plt
    from data_utils import apply_codec

    codecs = [("uncompressed", None), ("mp3", 32), ("aac", 32), ("opus", 16)]
    wav, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)

    fig, axes = plt.subplots(1, len(codecs), figsize=(4 * len(codecs), 4))
    for ax, (codec, br) in zip(axes, codecs):
        c = apply_codec(wav, sr, codec, br)
        db = 10 * torch.log10(mel(c) + 1e-9)
        ax.imshow(db[0].numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(codec if codec == "uncompressed" else f"{codec} {br}k")
        ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")

    plt.suptitle("Spectral Effect of Codec Compression")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Spectral diff saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--asv_root",      default=None)
    p.add_argument("--output_dir",    default="./results")
    p.add_argument("--split",         default="eval")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--tsne",          action="store_true")
    p.add_argument("--spectral",      default=None, help="Path to a .wav file")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model import CodecRobustDetector
    from data_utils import N_CODEC_CLASSES
    model = CodecRobustDetector(n_codec_classes=N_CODEC_CLASSES)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.asv_root:
        df = per_codec_eval(model, args.asv_root, device,
                             split=args.split, batch_size=args.batch_size)
        csv = out_dir / "per_codec_results.csv"
        df.to_csv(csv, index=False)
        print(df.to_string(index=False))
        print(f"\nSaved → {csv}")

    if args.tsne and args.asv_root:
        from data_utils import ASVspoof2019Dataset
        from torch.utils.data import DataLoader
        ds     = ASVspoof2019Dataset(args.asv_root, split=args.split, random_codec=True)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        plot_tsne(model, loader, device, str(out_dir / "tsne.png"))

    if args.spectral:
        plot_spectral_diff(args.spectral, str(out_dir / "spectral_diff.png"))


if __name__ == "__main__":
    main()