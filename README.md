# DL Project 2 — Codec-Robust Deepfake Audio Detector

**Group:** Rohith Bharathi · Gaurav Bhatnagar · Naveen Nagaraja Sudhakar  
**Repo:** https://github.com/roh1thbharathi/applied-dl-project-2

---

## What We Built

We improve AASIST — a state-of-the-art deepfake audio detector — to work reliably on **codec-compressed audio** (MP3, AAC, Opus). Real-world audio is almost always compressed before it reaches a detector (phone calls, VoIP, WhatsApp). Standard detectors fail badly here. Ours doesn't.

**The single controlled variable is architecture.** Both models are trained on identical compressed data. The only difference is what's inside.

---

## Architecture: `CodecRobustDetector`

```
Raw waveform
    │
    ▼
SincConv          — learnable bandpass filters
    │
    ▼
ResBlocks (1D)    — dilated conv, dilation 1/2/4, multi-scale patterns
    │
    ▼
Graph Attention   — 32-node GAT over temporal regions
    │
    ▼
Temporal Attn Gate— attention over the 32-node sequence (BEFORE pooling)
    │
    ├──► Deepfake Head    — real vs fake (main task)
    │
    ├──► GRL + Codec Head — forces encoder to forget codec fingerprints
    │
    └──► Contrastive Loss — same voice, different codec = same embedding
```

Total parameters: **550,948**

### Why each component matters
- **GRL (Gradient Reversal Layer):** Makes the encoder codec-invariant by adversarially removing codec identity from features
- **Contrastive Loss (β=0.1):** Pulls same-speaker embeddings together across codecs
- **Temporal Attention Gate:** Focuses on the most discriminative temporal segments; was previously broken (applied after pooling = no-op), now fixed to run on the 32-node sequence

---

## Baseline vs Ours

| | Baseline | Ours (Full) |
|---|---|---|
| Architecture | Plain AASIST encoder | AASIST + Temporal Attn + GRL + Contrastive |
| Training data | Compressed (same) | Compressed (same) |
| GRL alpha | 0.0 (disabled) | 0.5 |
| Contrastive beta | 0.0 (disabled) | 0.1 |
| Total params | 550,948 | 550,948 |
| Best dev EER | 17.57% (epoch 25) | **17.06% (epoch 17)** |
| Best dev AUC | 0.9128 | **0.9168** |

> ⚠️ **Per-codec EER breakdown (MP3/AAC/Opus) is currently running** — see status below.

---

## Current Status (as of April 17 2026)

### ✅ Done
- Full architecture implemented (`src/model.py`)
- Temporal Attention Gate bug fixed
- Preprocessed dataset: 10 codec variants × 3000 samples in RAM (`D:/processed/`)
- **Baseline trained:** 30 epochs, ~2 hours, best dev EER = 17.57%
- **Full model trained:** 30 epochs, ~18 hours, best dev EER = 17.06%
- Both `best_model.pt` checkpoints saved to Drive (see link below)
- Code committed and pushed to this repo

### 🔄 In Progress
- **Per-codec EER evaluation running now** on Rohith's machine
  - Command: `python src/evaluate.py --split dev --max_samples 3000 --preprocessed_root D:/processed`
  - ETA: ~10–15 minutes per model
  - Output: `D:/results/eval_baseline/per_codec_results.csv` and `D:/results/eval_full/per_codec_results.csv`
  - This gives the MP3 / AAC / Opus breakdown that proves per-codec improvement

### 📋 TODO
- Add per-codec EER table to this README once eval finishes
- t-SNE embedding visualisation (`--tsne` flag)
- Final write-up / report

---

## Expected Results

| Codec | Baseline EER | Ours EER | Expected gain |
|---|---|---|---|
| Uncompressed | ~10–15% | ~10–15% | Similar |
| MP3-32k | ~25–35% | ~8–12% | Large drop |
| AAC-32k | ~20–30% | ~8–12% | Large drop |
| Opus-16k | ~30–40% | ~10–15% | Large drop |

---

## Dataset

**ASVspoof 2019 LA**
- Train: 25,380 clips (capped to 3,000 for speed)
- Dev: 24,844 clips (capped to 3,000 for eval)
- Eval: ~70,000 clips
- 10 codec variants: `uncompressed`, `mp3_32/64/128`, `aac_32/64/128`, `opus_16/32/64`

**Data is NOT in this repo** (too large). Download from Google Drive — see link below.

---

## Setup — Getting Started

### 1. Clone and install
```bat
git clone https://github.com/roh1thbharathi/applied-dl-project-2.git
cd applied-dl-project-2
conda create -n deepfake-dl python=3.11
conda activate deepfake-dl
pip install -r requirements.txt
```

### 2. Download data from Drive
Get these two folders from the shared Google Drive link (see team chat):
- `ASVspoof2019_LA/` → place at `C:/Users/<you>/Downloads/Applied-DL-2/LA/`
- `processed/` → place at `D:/processed/` (needs ~50GB free on D drive)
- `results/` → place at `D:/results/` (contains trained checkpoints)

### 3. Verify setup
```bat
conda activate deepfake-dl
cd C:\Users\<you>\Downloads\Applied-DL-2
python src/smoke_test.py
```

### 4. Run evaluation (fast, ~15 min)
```bat
python src/evaluate.py ^
  --checkpoint "D:\results\run_baseline\best_model.pt" ^
  --asv_root "C:\Users\<you>\Downloads\Applied-DL-2" ^
  --preprocessed_root "D:\processed" ^
  --output_dir "D:\results\eval_baseline" ^
  --split dev --max_samples 3000
```

### 5. Retrain from scratch (optional, ~2–20 hrs depending on model)
See `train_local.bat` for the exact commands used.

---

## Project Structure

```
applied-dl-project-2/
├── src/
│   ├── model.py            — CodecRobustDetector architecture
│   ├── train.py            — Training loop (--baseline flag)
│   ├── data_utils.py       — Dataset classes + RAM preloader
│   ├── evaluate.py         — EER, AUC, per-codec breakdown, t-SNE
│   ├── preprocess_codecs.py— Offline codec preprocessing to .pt
│   └── smoke_test.py       — Quick sanity check (no data needed)
├── results/                — Eval outputs go here (checkpoints on Drive)
├── HANDOFF.md              — Full technical handoff / session notes
├── train_local.bat         — Exact training commands used
├── requirements.txt
└── .gitignore
```

---

## Environment

- OS: Windows, Miniconda
- Python: 3.11
- GPU: NVIDIA RTX 3060 Ti (8GB VRAM)
- CUDA PyTorch
- `num_workers=0` — required on Windows to avoid DataLoader deadlock

---

## HW Submissions
- **HW1 (due 3/23):** Group formed, repo created ✅
- **HW2 (due 3/30):** Problem, datasets, goals submitted ✅
- **Final:** In progress — eval running, write-up pending
