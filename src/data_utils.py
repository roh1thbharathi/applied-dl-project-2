"""
Data utilities: ASVspoof 2019 + WaveFake loaders + codec augmentation pipeline.
"""

import os
import random
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, List

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

CODEC_TO_IDX  = {v: i for i, v in enumerate(_CODEC_CLASSES)}
N_CODEC_CLASSES = len(_CODEC_CLASSES)   # 10


def sample_random_codec():
    codec   = random.choice(list(CODEC_CONFIG.keys()))
    bitrate = random.choice(CODEC_CONFIG[codec])
    return codec, bitrate


# ============================================================
# Codec compression
# ============================================================

def apply_codec(waveform, sample_rate, codec, bitrate=None, use_ffmpeg=False):
    """Apply lossy codec to waveform tensor. Falls back gracefully on failure."""
    if codec == "uncompressed":
        return waveform

    if use_ffmpeg:
        return _apply_ffmpeg(waveform, sample_rate, codec, bitrate)
    else:
        return _apply_torchaudio(waveform, sample_rate, codec, bitrate)


def _apply_torchaudio(waveform, sample_rate, codec, bitrate):
    fmt_map = {"mp3": "mp3", "aac": "adts", "opus": "opus"}
    fmt = fmt_map.get(codec)
    if not fmt:
        return waveform
    try:
        out = torchaudio.functional.apply_codec(
            waveform, sample_rate,
            format=fmt,
            compression=float(bitrate) if bitrate else -4.0,
        )
        T = waveform.shape[-1]
        out = out[:, :T] if out.shape[-1] >= T else F_pad(out, T)
        return out
    except Exception:
        return waveform


def _apply_ffmpeg(waveform, sample_rate, codec, bitrate):
    fmt_map = {"mp3": "mp3", "aac": "adts", "opus": "opus"}
    fmt = fmt_map.get(codec)
    if not fmt:
        return waveform
    with tempfile.TemporaryDirectory() as tmp:
        in_p  = os.path.join(tmp, "in.wav")
        out_p = os.path.join(tmp, f"out.{fmt}")
        torchaudio.save(in_p, waveform, sample_rate)
        cmd = ["ffmpeg", "-y", "-i", in_p]
        if bitrate:
            cmd += ["-b:a", f"{bitrate}k"]
        cmd.append(out_p)
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            out, _ = torchaudio.load(out_p)
        except Exception:
            return waveform
    T = waveform.shape[-1]
    out = out[:, :T] if out.shape[-1] >= T else F_pad(out, T)
    return out


def F_pad(t, target_len):
    return torch.nn.functional.pad(t, (0, target_len - t.shape[-1]))


# ============================================================
# Shared audio loading helper
# ============================================================

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
    return wav   # (1, max_len)


# ============================================================
# ASVspoof 2019 LA
# ============================================================

class ASVspoof2019Dataset(Dataset):
    """
    root/
      LA/
        ASVspoof2019_LA_train/flac/*.flac
        ASVspoof2019_LA_dev/flac/*.flac
        ASVspoof2019_LA_eval/flac/*.flac
        ASVspoof2019_LA_cm_protocols/
          ASVspoof2019.LA.cm.train.trn.txt
          ASVspoof2019.LA.cm.dev.trl.txt
          ASVspoof2019.LA.cm.eval.trl.txt

    Label: 0 = bonafide (real), 1 = spoof (fake)
    """

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

        if self.random_codec:
            codec, bitrate = sample_random_codec()
        else:
            codec, bitrate = (self.codec or "uncompressed"), self.bitrate

        codec_idx = CODEC_TO_IDX.get((codec, bitrate), 0)
        wav_c     = apply_codec(wav, self.target_sr, codec, bitrate)

        if self.contrastive:
            c2, b2    = sample_random_codec()
            wav_c2    = apply_codec(wav, self.target_sr, c2, b2)
            return {
                "waveform":   wav_c.squeeze(0),
                "waveform2":  wav_c2.squeeze(0),
                "label":      torch.tensor(label, dtype=torch.long),
                "codec_idx":  torch.tensor(codec_idx, dtype=torch.long),
                "codec_idx2": torch.tensor(CODEC_TO_IDX.get((c2, b2), 0), dtype=torch.long),
            }

        return {
            "waveform":  wav_c.squeeze(0),
            "label":     torch.tensor(label, dtype=torch.long),
            "codec_idx": torch.tensor(codec_idx, dtype=torch.long),
        }


# ============================================================
# WaveFake
# ============================================================

class WaveFakeDataset(Dataset):
    """
    root/
      real/              <- bonafide clips (.wav)
      ljspeech_hifiGAN/  <- fake clips (.wav)
      ljspeech_melgan/
      ljspeech_waveglow/
      ... (any vocoder folder)

    Label: 0 = real, 1 = fake
    """

    def __init__(self, root, target_sr=16000, max_len_sec=4.0,
                 random_codec=True, contrastive=False):
        self.root         = Path(root)
        self.target_sr    = target_sr
        self.max_len      = int(target_sr * max_len_sec)
        self.random_codec = random_codec
        self.contrastive  = contrastive

        self.items = []
        real_dir = self.root / "real"
        if real_dir.exists():
            for f in real_dir.glob("*.wav"):
                self.items.append((f, 0))
        for d in self.root.iterdir():
            if d.is_dir() and d.name != "real":
                for f in d.glob("*.wav"):
                    self.items.append((f, 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav         = load_audio(path, self.target_sr, self.max_len)
        codec, br   = sample_random_codec() if self.random_codec else ("uncompressed", None)
        codec_idx   = CODEC_TO_IDX.get((codec, br), 0)
        wav_c       = apply_codec(wav, self.target_sr, codec, br)

        if self.contrastive:
            c2, b2  = sample_random_codec()
            wav_c2  = apply_codec(load_audio(path, self.target_sr, self.max_len),
                                  self.target_sr, c2, b2)
            return {
                "waveform":   wav_c.squeeze(0),
                "waveform2":  wav_c2.squeeze(0),
                "label":      torch.tensor(label, dtype=torch.long),
                "codec_idx":  torch.tensor(codec_idx, dtype=torch.long),
                "codec_idx2": torch.tensor(CODEC_TO_IDX.get((c2, b2), 0), dtype=torch.long),
            }

        return {
            "waveform":  wav_c.squeeze(0),
            "label":     torch.tensor(label, dtype=torch.long),
            "codec_idx": torch.tensor(codec_idx, dtype=torch.long),
        }


# ============================================================
# DataLoader factory
# ============================================================

def get_dataloaders(asv_root=None, wavefake_root=None, batch_size=32,
                    num_workers=4, random_codec=True, contrastive=False,
                    target_sr=16000, max_len_sec=4.0):
    from torch.utils.data import ConcatDataset

    buckets = {"train": [], "dev": [], "eval": []}

    if asv_root:
        for split in ["train", "dev", "eval"]:
            buckets[split].append(
                ASVspoof2019Dataset(asv_root, split=split, target_sr=target_sr,
                                    max_len_sec=max_len_sec, random_codec=random_codec,
                                    contrastive=contrastive)
            )

    if wavefake_root:
        ds = WaveFakeDataset(wavefake_root, target_sr=target_sr,
                             max_len_sec=max_len_sec, random_codec=random_codec,
                             contrastive=contrastive)
        n  = len(ds)
        splits = [int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)]
        tr, dv, ev = torch.utils.data.random_split(
            ds, splits, generator=torch.Generator().manual_seed(42))
        buckets["train"].append(tr)
        buckets["dev"].append(dv)
        buckets["eval"].append(ev)

    loaders = {}
    for split, lst in buckets.items():
        if not lst:
            continue
        combined = ConcatDataset(lst) if len(lst) > 1 else lst[0]
        loaders[split] = DataLoader(
            combined, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        )
    return loaders


if __name__ == "__main__":
    print(f"Codec classes: {N_CODEC_CLASSES}")
    for pair, idx in CODEC_TO_IDX.items():
        print(f"  [{idx}] {pair[0]:12s} {str(pair[1]):5s} kbps")