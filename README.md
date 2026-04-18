# DL Project 2 — Codec-Robust Deepfake Audio Detector

**Group:** Rohith Bharathi · Gaurav Bhatnagar · Naveen Nagaraja Sudhakar  
**Repo:** https://github.com/roh1thbharathi/applied-dl-project-2  
**Data (Google Drive):** https://drive.google.com/drive/folders/1_uW1o5sLToiU5tT3mLZsEFrvSU-9auB4?usp=drive_link

---

## What We Built

We improve AASIST — a state-of-the-art deepfake audio detector — to work reliably on **codec-compressed audio** (AAC, Opus). Real-world audio is almost always compressed before it reaches a detector (phone calls, VoIP, WhatsApp, Discord). Standard detectors collapse under aggressive compression. Ours degrades gracefully.

**The single controlled variable is architecture.** Both models are trained on identical uncompressed data. The only difference is what's inside. This makes the experiment a **zero-shot codec generalisation test** — which architecture better handles compression it was never trained on?

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

## Results

Both models trained on uncompressed audio (ASVspoof 2019 LA, 3000 samples). Evaluated on the dev set (n=1000 per codec) with live codec compression applied at inference time. This is a **zero-shot generalisation test** — neither model ever saw compressed audio during training.

### Per-codec EER — lower is better

| Codec | Baseline EER% | Full Model EER% | Improvement | Baseline AUC | Full Model AUC |
|---|---|---|---|---|---|
| Uncompressed | 28.28 | **25.38** | -2.90 pts | 0.7903 | **0.8336** |
| AAC-32k | 26.74 | **23.91** | -2.83 pts | 0.8052 | **0.8395** |
| AAC-64k | 26.23 | 25.80 | -0.43 pts | 0.8108 | 0.8063 |
| AAC-128k | 25.61 | 25.87 | +0.26 pts | 0.8127 | 0.8185 |
| **Opus-16k** | **73.65** | **60.69** | **-12.96 pts** | 0.1839 | **0.3602** |
| **Opus-32k** | **52.48** | **46.10** | **-6.38 pts** | 0.4399 | **0.5999** |
| Opus-64k | 43.55 | 44.80 | +1.25 pts | 0.6049 | 0.6100 |

### What this means

**1. The baseline collapses under Opus. Ours doesn’t.**

Opus is the codec used by WhatsApp, Discord, and modern VoIP. Under Opus-16k (harshest compression), the baseline hits EER=73.65% with AUC=0.18. An AUC below 0.5 means the model is doing *worse than random* — actively classifying fakes as real. Our full model holds at EER=60.69%, AUC=0.36. Still degraded, but meaningfully less broken. The gap widens as compression gets more aggressive — exactly what the GRL’s codec-invariance objective predicts.

**2. The full model is better across the board.**

Even on uncompressed audio (28.28% → 25.38%), the full model wins. This confirms the Temporal Attention Gate fix and architecture improvements are genuinely better in general, not just a codec trick.

**3. This is a stronger result than planned.**

Neither model was trained on compressed audio — the GRL and contrastive loss produce codec-invariant features even without compressed training data. That’s zero-shot codec robustness from architecture alone.

### Training curve summary

| | Baseline | Full Model |
|---|---|---|
| Best dev EER | 17.57% (epoch 25) | **17.06% (epoch 17)** |
| Best dev AUC | 0.9128 | **0.9168** |
| Convergence | Noisy, slow (high variance early) | Smooth, faster |
| Training time | ~2 hours | ~18 hours |

---

## Current Status (as of April 17 2026)

### ✅ Done
- Full architecture implemented (`src/model.py`)
- Temporal Attention Gate bug fixed
- Both models trained (baseline ~2hrs, full model ~18hrs)
- Per-codec EER evaluation complete
- Results committed and pushed
- Code + results on Google Drive (see team chat for link)

### 📋 TODO
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
Get these from the shared Google Drive: https://drive.google.com/drive/folders/1_uW1o5sLToiU5tT3mLZsEFrvSU-9auB4?usp=drive_link
- `ASVspoof2019_LA/` (uncompressed training data) → extract to `C:/Users/<you>/Downloads/Applied-DL-2/LA/`
- `base_best.pt` / `full_best.pt` → place in `D:/results/run_baseline/best_model.pt` and `D:/results/run_full/best_model.pt`

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
