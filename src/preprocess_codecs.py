"""
Preprocess: apply all codec variants to ASVspoof clips and save to disk.
Run this ONCE before training. Speeds up training 5-10x by eliminating
live codec augmentation bottleneck.

Usage:
    python src/preprocess_codecs.py --asv_root "C:\\path\\to\\Applied-DL-2"

Output structure:
    data/processed/
        uncompressed/  <- just copies of originals as .pt tensors
        mp3_32/
        mp3_64/
        mp3_128/
        aac_32/
        aac_64/
        aac_128/
        opus_16/
        opus_32/
        opus_64/
    Each file: <filename>.pt  containing tensor of shape (32000,) = 2s @ 16kHz
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, "src")
from data_utils import CODEC_CONFIG, apply_codec, load_audio

TARGET_SR  = 16000
MAX_LEN    = 32000   # 2 seconds


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asv_root", required=True,
                   help="Project root containing LA/ folder")
    p.add_argument("--splits", nargs="+",
                   default=["train", "dev", "eval"])
    p.add_argument("--out_dir", default=None,
                   help="Where to write .pt files (default: <asv_root>/data/processed)")
    return p.parse_args()


def process_split(split, asv_root, out_dir):
    split_map = {
        "train": "ASVspoof2019_LA_train/flac",
        "dev":   "ASVspoof2019_LA_dev/flac",
        "eval":  "ASVspoof2019_LA_eval/flac",
    }
    flac_dir = Path(asv_root) / "LA" / split_map[split]
    files    = list(flac_dir.glob("*.flac"))
    print(f"\n[{split}] {len(files)} files found in {flac_dir}")

    for codec, bitrates in CODEC_CONFIG.items():
        for bitrate in bitrates:
            tag     = f"{codec}_{bitrate}" if bitrate else "uncompressed"
            dst_dir = out_dir / tag / split
            dst_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(dst_dir.glob("*.pt")))
            if existing == len(files):
                print(f"  {tag:15s} already done ({existing} files), skipping.")
                continue

            print(f"  {tag:15s} processing...")
            for fpath in tqdm(files, desc=f"  {tag}", leave=False):
                dst = dst_dir / (fpath.stem + ".pt")
                if dst.exists():
                    continue
                try:
                    wav = load_audio(fpath, TARGET_SR, MAX_LEN)   # (1, 32000)
                    wav = apply_codec(wav, TARGET_SR, codec, bitrate)
                    torch.save(wav.squeeze(0), dst)                 # (32000,)
                except Exception as e:
                    print(f"    WARN: {fpath.name} failed: {e}")


def main():
    args    = get_args()
    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(args.asv_root) / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Codec Preprocessing")
    print(f"  Output : {out_dir}")
    print(f"  Splits : {args.splits}")
    print(f"  Codecs : {sum(len(v) for v in CODEC_CONFIG.values())} variants")
    print("=" * 55)

    for split in args.splits:
        process_split(split, args.asv_root, out_dir)

    print("\nDone! Now run training with --preprocessed flag (coming soon)")
    print(f"Processed data at: {out_dir}")


if __name__ == "__main__":
    main()
