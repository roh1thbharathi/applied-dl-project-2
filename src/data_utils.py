"""
Data utilities: ASVspoof 2019 loaders.

Two modes:
  PREPROCESSED — loads .pt files from disk, preloads subset into RAM at startup
  LIVE         — applies codec on-the-fly (smoke test only)
"""

import os
import random
import tempfile
import subprocess
from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Codec config  (10 classes total)
# ============================================================

CODEC_CONFIG = {
    "uncompressed": [None],
    "mp3":  [32, 64, 128],
    "aac":  [32, 64, 128],
    "opus": [16, 32, 64],
}

_CODEC_CLASSES = []
for _c, _brs in CODEC_CONFIG.items():
    for _b in _brs:
        _CODEC_CLASSES.append((_c, _b))

CODEC_TO_IDX    = {v: i for i, v in enumerate(_CODEC_CLASSES)}
N_CODEC_CLASSES = len(_CODEC_CLASSES)   # 10

def codec_tag(codec, bitrate):
    return f"{codec}_{bitrate}" if bitrate else "uncompressed"

ALL_CODEC_TAGS = [codec_tag(c, b) for c, brs in CODEC_CONFIG.items() for b in brs]


def sample_random_codec():
    codec   = random.choice(list(CODEC_CONFIG.keys()))
    bitrate = random.choice(CODEC_CONFIG[codec])
    return codec, bitrate


# ============================================================
# Live codec (smoke test only)
# ============================================================

_effector_cache: dict = {}

def _get_effector(codec, bitrate):
    from torchaudio.io import AudioEffector, CodecConfig
    key = (codec, bitrate)
    if key not in _effector_cache:
        br = int(bitrate) * 1000 if bitrate else 128_000
        if codec == "mp3":
            _effector_cache[key] = AudioEffector(format="mp3",
                codec_config=CodecConfig(bit_rate=br))
        elif codec == "aac":
            _effector_cache[key] = AudioEffector(format="adts", encoder="aac",
                codec_config=CodecConfig(bit_rate=br))
        elif codec == "opus":
            _effector_cache[key] = AudioEffector(format="ogg", encoder="libopus",
                codec_config=CodecConfig(bit_rate=br))
        else:
            return None
    return _effector_cache[key]


def apply_codec(waveform, sample_rate, codec, bitrate=None):
    if codec == "uncompressed":
        return waveform
    try:
        effector = _get_effector(codec, bitrate)
        if effector is None:
            return waveform
        wav_cpu = waveform.squeeze(0).unsqueeze(-1).cpu()
        out     = effector.apply(wav_cpu, sample_rate)
        out     = out.squeeze(-1).unsqueeze(0)
        T       = waveform.shape[-1]
        return out[:, :T] if out.shape[-1] >= T else torch.nn.functional.pad(
            out, (0, T - out.shape[-1]))
    except Exception:
        return waveform


def load_audio(path, target_sr=16000, max_len=64000):
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    T = wav.shape[-1]
    if T >= max_len:
        wav = wav[:, :max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - T))
    return wav


# ============================================================
# ASVspoof 2019 — PREPROCESSED (fast, use for training)
# ============================================================

class ASVspoof2019Preprocessed(Dataset):
    """
    Loads pre-cached .pt files. On first use, preloads ALL codec variants
    for the selected subset into RAM so every epoch after is pure RAM access.

    processed_root/
        mp3_32/train/<filename>.pt
        opus_16/dev/<filename>.pt
        ...
    """

    PROTO = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    def __init__(self, asv_root, processed_root, split="train", contrastive=False):
        self.processed_root = Path(processed_root)
        self.split          = split
        self.contrastive    = contrastive
        self._ram           = {}   # fname -> {tag -> tensor}  (all codecs preloaded)

        proto_path = (Path(asv_root) / "LA" /
                      "ASVspoof2019_LA_cm_protocols" / self.PROTO[split])
        df = pd.read_csv(proto_path, sep=" ", header=None,
                         names=["speaker", "filename", "dash", "system_id", "label"])
        self.records = df[["filename", "label"]].reset_index(drop=True)

        self.available_tags = [
            t for t in ALL_CODEC_TAGS
            if (self.processed_root / t / split).exists()
        ]
        assert self.available_tags, (
            f"No preprocessed folders found under {processed_root}. "
            f"Run: python src/preprocess_codecs.py --asv_root <root>")

    def preload_to_ram(self, indices):
        """
        Load ALL codec variants for the given record indices into RAM.
        Called once by get_dataloaders after subsetting.
        After this returns, HDD is never touched again.
        """
        filenames = [self.records.iloc[i]["filename"] for i in indices]
        total     = len(filenames)
        print(f"  [RAM] Preloading {total} clips \u00d7 "
              f"{len(self.available_tags)} codecs into RAM...")
        for i, fname in enumerate(filenames):
            self._ram[fname] = {}
            for tag in self.available_tags:
                pt = self.processed_root / tag / self.split / f"{fname}.pt"
                if pt.exists():
                    try:
                        self._ram[fname][tag] = torch.load(pt, weights_only=True)
                    except Exception:
                        pass   # skip corrupted file
            if (i + 1) % 1000 == 0:
                print(f"  [RAM] {i+1}/{total}...")
        print(f"  [RAM] Done. HDD reads = 0 from now on.")

    def _get(self, fname, tag):
        """Get tensor from RAM, always compressed."""
        if fname in self._ram:
            # return requested tag or any available one
            if tag in self._ram[fname]:
                return self._ram[fname][tag]
            if self._ram[fname]:
                return random.choice(list(self._ram[fname].values()))
        # fallback disk read (only happens before preload or for eval)
        for t in [tag] + [x for x in self.available_tags if x != tag]:
            pt = self.processed_root / t / self.split / f"{fname}.pt"
            if pt.exists():
                try:
                    return torch.load(pt, weights_only=True)
                except Exception:
                    continue
        raise FileNotFoundError(f"No valid codec file for {fname}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        fname = row["filename"]
        label = 0 if row["label"] == "bonafide" else 1

        tag       = random.choice(self.available_tags)
        codec, br = next((c, b) for c, brs in CODEC_CONFIG.items()
                         for b in brs if codec_tag(c, b) == tag)
        wav       = self._get(fname, tag)
        codec_idx = CODEC_TO_IDX.get((codec, br), 0)

        if self.contrastive:
            other = [t for t in self.available_tags if t != tag]
            tag2  = random.choice(other) if other else tag
            c2, b2 = next((c, b) for c, brs in CODEC_CONFIG.items()
                          for b in brs if codec_tag(c, b) == tag2)
            wav2  = self._get(fname, tag2)
            return {
                "waveform":   wav,
                "waveform2":  wav2,
                "label":      torch.tensor(label,  dtype=torch.long),
                "codec_idx":  torch.tensor(codec_idx, dtype=torch.long),
                "codec_idx2": torch.tensor(CODEC_TO_IDX.get((c2, b2), 0),
                                           dtype=torch.long),
            }

        return {
            "waveform":  wav,
            "label":     torch.tensor(label,     dtype=torch.long),
            "codec_idx": torch.tensor(codec_idx, dtype=torch.long),
        }


# ============================================================
# ASVspoof 2019 — LIVE (smoke test only, very slow)
# ============================================================

class ASVspoof2019Dataset(Dataset):
    SPLITS = {
        "train": ("ASVspoof2019_LA_train/flac", "ASVspoof2019.LA.cm.train.trn.txt"),
        "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019.LA.cm.dev.trl.txt"),
        "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019.LA.cm.eval.trl.txt"),
    }

    def __init__(self, root, split="train", target_sr=16000, max_len_sec=4.0,
                 codec=None, bitrate=None, random_codec=True, contrastive=False):
        self.root         = Path(root) / "LA"
        self.target_sr    = target_sr
        self.max_len      = int(target_sr * max_len_sec)
        self.codec        = codec
        self.bitrate      = bitrate
        self.random_codec = random_codec
        self.contrastive  = contrastive

        audio_dir, proto_file = self.SPLITS[split]
        self.audio_dir = self.root / audio_dir
        proto_path     = self.root / "ASVspoof2019_LA_cm_protocols" / proto_file
        df = pd.read_csv(proto_path, sep=" ", header=None,
                         names=["speaker", "filename", "dash", "system_id", "label"])
        self.records = df[["filename", "label"]].reset_index(drop=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        wav   = load_audio(self.audio_dir / f"{row['filename']}.flac",
                           self.target_sr, self.max_len)
        label = 0 if row["label"] == "bonafide" else 1
        codec, bitrate = (sample_random_codec() if self.random_codec
                          else ((self.codec or "uncompressed"), self.bitrate))
        codec_idx = CODEC_TO_IDX.get((codec, bitrate), 0)
        wav_c     = apply_codec(wav, self.target_sr, codec, bitrate)

        if self.contrastive:
            c2, b2 = sample_random_codec()
            wav_c2 = apply_codec(wav, self.target_sr, c2, b2)
            return {
                "waveform":   wav_c.squeeze(0),
                "waveform2":  wav_c2.squeeze(0),
                "label":      torch.tensor(label,     dtype=torch.long),
                "codec_idx":  torch.tensor(codec_idx, dtype=torch.long),
                "codec_idx2": torch.tensor(CODEC_TO_IDX.get((c2, b2), 0),
                                           dtype=torch.long),
            }
        return {
            "waveform":  wav_c.squeeze(0),
            "label":     torch.tensor(label,     dtype=torch.long),
            "codec_idx": torch.tensor(codec_idx, dtype=torch.long),
        }


# ============================================================
# DataLoader factory
# ============================================================

def get_dataloaders(asv_root=None, wavefake_root=None, batch_size=32,
                    num_workers=4, random_codec=True, contrastive=False,
                    target_sr=16000, max_len_sec=4.0,
                    preprocessed_root=None, max_samples=None):
    from torch.utils.data import ConcatDataset, Subset

    buckets = {"train": [], "dev": [], "eval": []}

    if asv_root:
        if preprocessed_root and Path(preprocessed_root).exists():
            print(f"  [DataLoader] Using preprocessed data from: {preprocessed_root}")
            for split in ["train", "dev", "eval"]:
                try:
                    ds = ASVspoof2019Preprocessed(
                        asv_root=asv_root,
                        processed_root=preprocessed_root,
                        split=split,
                        contrastive=contrastive,
                    )
                    print(f"  [DataLoader] {split:5s}: {len(ds):,} samples "
                          f"| {len(ds.available_tags)} codec variants available")
                    buckets[split].append(ds)
                except Exception as e:
                    print(f"  [DataLoader] WARN: could not load {split}: {e}")
        else:
            print("  [DataLoader] WARNING: No preprocessed data — using live augmentation (slow).")
            for split in ["train", "dev", "eval"]:
                buckets[split].append(
                    ASVspoof2019Dataset(asv_root, split=split, target_sr=target_sr,
                                       max_len_sec=max_len_sec,
                                       random_codec=random_codec,
                                       contrastive=contrastive))

    loaders = {}
    for split, lst in buckets.items():
        if not lst:
            continue
        combined = ConcatDataset(lst) if len(lst) > 1 else lst[0]

        # Cap dataset size
        if max_samples and len(combined) > max_samples:
            indices  = torch.randperm(len(combined))[:max_samples].tolist()
            combined = Subset(combined, indices)
            print(f"  [DataLoader] {split} capped at {max_samples} samples")

            # Preload ALL codec variants for capped samples into RAM
            # so every epoch after startup is pure RAM — no HDD
            if preprocessed_root:
                ds_ref = combined.dataset if hasattr(combined, "dataset") else combined
                if hasattr(ds_ref, "preload_to_ram"):
                    ds_ref.preload_to_ram(combined.indices)

        loaders[split] = DataLoader(
            combined, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,          # 0 = no worker processes (Windows safe)
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=False,
        )
    return loaders


if __name__ == "__main__":
    print(f"Codec classes: {N_CODEC_CLASSES}")
    for pair, idx in CODEC_TO_IDX.items():
        print(f"  [{idx}] {pair[0]:12s} {str(pair[1]):5s} kbps")
