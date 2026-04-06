"""
Training script.

Usage:
  python src/train.py \
    --asv_root    /path/to/ASVspoof2019 \
    --wavefake_root /path/to/WaveFake \
    --epochs 30 --batch_size 32 \
    --alpha 0.5 --beta 0.1
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CodecRobustDetector, ContrastiveLoss
from data_utils import get_dataloaders, N_CODEC_CLASSES
from evaluate import compute_eer, compute_auc


# ── GRL lambda schedule (Ganin et al. 2016) ─────────────────────────────────

def grl_lambda(step, total_steps, lam_max=1.0):
    p = step / max(total_steps, 1)
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ── Train one epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, df_crit, codec_crit, con_crit,
                alpha, beta, step, total_steps, device, writer, use_contrastive):
    model.train()
    totals = {"total": 0, "df": 0, "codec": 0, "con": 0}
    n = 0

    for batch in loader:
        lam = grl_lambda(step, total_steps)
        model.set_lambda(lam)

        wf        = batch["waveform"].to(device)
        labels    = batch["label"].to(device)
        codec_idx = batch["codec_idx"].to(device)

        out    = model(wf)
        l_df   = df_crit(out["deepfake_logits"], labels)
        l_codec = codec_crit(out["codec_logits"], codec_idx)
        loss   = l_df + alpha * l_codec

        l_con_val = 0.0
        if use_contrastive and "waveform2" in batch:
            wf2       = batch["waveform2"].to(device)
            ci2       = batch["codec_idx2"].to(device)
            out2      = model(wf2)
            # interleave pairs: (2B, D)
            B   = wf.size(0)
            e1  = out["embedding"]
            e2  = out2["embedding"]
            pairs = torch.stack([e1, e2], dim=1).view(2 * B, -1)
            l_con = con_crit(pairs)
            l_con_val = l_con.item()
            loss = loss + alpha * codec_crit(out2["codec_logits"], ci2) + beta * l_con

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        totals["total"] += loss.item()
        totals["df"]    += l_df.item()
        totals["codec"] += l_codec.item()
        totals["con"]   += l_con_val
        n += 1; step += 1

        if step % 100 == 0:
            for k, v in totals.items():
                writer.add_scalar(f"train/{k}", v / n, step)
            writer.add_scalar("train/lambda", lam, step)

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    print(f"  train | total={avg['total']:.4f}  df={avg['df']:.4f}  "
          f"codec={avg['codec']:.4f}  con={avg['con']:.4f}  λ={lam:.3f}")
    return step


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, split="dev"):
    model.eval()
    scores_all, labels_all = [], []
    for batch in loader:
        out = model(batch["waveform"].to(device))
        s   = torch.softmax(out["deepfake_logits"], -1)[:, 1]
        scores_all.append(s.cpu())
        labels_all.append(batch["label"])
    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()
    eer = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    print(f"  {split:5s} | EER={eer*100:.2f}%  AUC={auc:.4f}")
    return {"eer": eer, "auc": auc}


# ── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asv_root",        type=str,   default=None)
    p.add_argument("--wavefake_root",   type=str,   default=None)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="Codec domain loss weight")
    p.add_argument("--beta",            type=float, default=0.1,
                   help="Contrastive loss weight")
    p.add_argument("--embed_dim",       type=int,   default=256)
    p.add_argument("--lam_max",         type=float, default=1.0)
    p.add_argument("--output_dir",      type=str,   default="./results")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--no_contrastive",  action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(str(out_dir / "tb_logs"))

    loaders = get_dataloaders(
        asv_root=args.asv_root, wavefake_root=args.wavefake_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        random_codec=True, contrastive=not args.no_contrastive,
    )
    assert "train" in loaders, "No training data."

    model = CodecRobustDetector(embed_dim=args.embed_dim,
                                n_codec_classes=N_CODEC_CLASSES).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20)

    df_crit    = nn.CrossEntropyLoss()
    codec_crit = nn.CrossEntropyLoss()
    con_crit   = ContrastiveLoss(temperature=0.07)

    total_steps = args.epochs * len(loaders["train"])
    step, best_eer = 0, 1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0   = time.time()
        step = train_epoch(
            model, loaders["train"], optimizer,
            df_crit, codec_crit, con_crit,
            args.alpha, args.beta, step, total_steps,
            device, writer, not args.no_contrastive,
        )
        scheduler.step()

        if "dev" in loaders:
            m = evaluate(model, loaders["dev"], device, "dev")
            writer.add_scalar("val/EER", m["eer"], epoch)
            writer.add_scalar("val/AUC", m["auc"], epoch)
            if m["eer"] < best_eer:
                best_eer = m["eer"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                print(f"  ✓ best EER so far: {best_eer*100:.2f}%")

        print(f"  epoch time: {time.time()-t0:.1f}s")

    if "eval" in loaders:
        print("\nFinal test evaluation:")
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        evaluate(model, loaders["eval"], device, "eval")

    writer.close()
    print("Done. Results in:", out_dir)


if __name__ == "__main__":
    main()