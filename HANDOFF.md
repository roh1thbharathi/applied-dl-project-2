# HANDOFF DOCUMENT — DL Project-2: Codec-Robust Deepfake Detector
# Paste this entire file at the start of the next conversation window

---

## PROJECT SUMMARY

Goal: Improve AASIST deepfake audio detector to work on codec-compressed audio (MP3/AAC/Opus).
Baseline: plain AASIST trained on compressed audio, tested on compressed audio → high EER
Ours: AASIST + Temporal Attention Gate + GRL + Contrastive Loss, trained on compressed → low EER
The only variable is the architecture — same training data, same test data.

---

## ENVIRONMENT

- OS: Windows, Miniconda
- Conda env: deepfake-dl (Python 3.11, CUDA PyTorch)
- GPU: NVIDIA RTX 3060 Ti (8GB VRAM)
- Project folder: C:\Users\Admin\Downloads\Applied-DL-2
- Preprocessed data: D:\processed (train + dev only, 10 codec variants, HDD)
- Results: D:\results\

To activate environment in any new terminal:
    conda activate deepfake-dl
    cd C:\Users\Admin\Downloads\Applied-DL-2

---

## ARCHITECTURE (src/model.py)

CodecRobustDetector:
  1. SincConv — learnable bandpass filters on raw waveform
  2. ResBlocks — dilated 1D conv (dilation 1,2,4) for multi-scale patterns
  3. Graph Attention (GAT) — 32-node attention over temporal regions
  4. Temporal Attention Gate — FIXED: runs on 32-node sequence BEFORE pooling (was bug)
  5. GRL + Codec Head — forces encoder to forget codec fingerprints
  6. Contrastive Loss — same voice, different codec = same embedding

Total params: 550,948

---

## TRAINING STATUS

Run 1 — BASELINE (currently running, 18/30 epochs done):
  Command:
    python src/train.py ^
      --asv_root "C:\Users\Admin\Downloads\Applied-DL-2" ^
      --preprocessed_root "D:\processed" ^
      --baseline ^
      --epochs 30 --batch_size 32 --max_len_sec 2.0 ^
      --lr 3e-4 --embed_dim 256 --num_workers 0 ^
      --max_samples 3000 ^
      --output_dir "D:\results\run_baseline"

  What --baseline does:
    - alpha=0.0 (GRL disabled)
    - beta=0.0 (contrastive disabled)
    - no_contrastive=True
    - Same compressed training data as full model
    - Plain AASIST encoder only

Run 2 — FULL MODEL (run after baseline finishes):
    python src/train.py ^
      --asv_root "C:\Users\Admin\Downloads\Applied-DL-2" ^
      --preprocessed_root "D:\processed" ^
      --epochs 30 --batch_size 32 --max_len_sec 2.0 ^
      --lr 3e-4 --alpha 0.5 --beta 0.1 --embed_dim 256 --num_workers 0 ^
      --max_samples 3000 ^
      --output_dir "D:\results\run_full"

---

## DATA

Dataset: ASVspoof 2019 LA
  - Train: 25,380 clips (capped to 3000 for speed)
  - Dev:   24,844 clips (capped to 3000 for speed)
  - Eval:  ~70,000 clips (live augmentation at end, runs once)

Preprocessed at D:\processed:
  Folders: uncompressed, mp3_32, mp3_64, mp3_128,
           aac_32, aac_64, aac_128, opus_16, opus_32, opus_64
  Each folder has train/ and dev/ subfolders with .pt tensors (32000 samples = 2s @ 16kHz)
  Eval was NOT preprocessed (disk ran out) — falls back to live augmentation

RAM caching:
  At startup, preload_to_ram() loads ALL 10 codec variants for 3000 samples into RAM
  After preload: HDD = 0 KB/s, GPU = ~98%, epoch time = 2-3 min

---

## KEY FILES

src/model.py     — Full architecture (CodecRobustDetector, ContrastiveLoss)
src/train.py     — Training loop (--baseline flag, --max_samples, --preprocessed_root)
src/data_utils.py — Dataset classes (ASVspoof2019Preprocessed with RAM cache)
src/evaluate.py  — EER, AUC, per-codec breakdown, t-SNE
src/smoke_test.py — Quick sanity check (no data needed)

---

## EVALUATION (after both runs finish)

Per-codec EER comparison:
    python src/evaluate.py ^
      --checkpoint "D:\results\run_baseline\best_model.pt" ^
      --asv_root "C:\Users\Admin\Downloads\Applied-DL-2" ^
      --output_dir "D:\results\eval_baseline" ^
      --split eval

    python src/evaluate.py ^
      --checkpoint "D:\results\run_full\best_model.pt" ^
      --asv_root "C:\Users\Admin\Downloads\Applied-DL-2" ^
      --output_dir "D:\results\eval_full" ^
      --split eval --tsne

---

## EXPECTED RESULTS TABLE

| Codec       | Baseline EER | Ours EER | Improvement |
|-------------|-------------|----------|-------------|
| Uncompressed| ~10-15%     | ~10-15%  | similar     |
| MP3-32k     | ~25-35%     | ~8-12%   | big drop    |
| AAC-32k     | ~20-30%     | ~8-12%   | big drop    |
| Opus-16k    | ~30-40%     | ~10-15%  | big drop    |

---

## KNOWN ISSUES / DECISIONS

1. 3000 samples used (not full 25k) due to HDD bottleneck
   - Fair comparison: both baseline and full use same 3000 samples
   - Numbers will be noisier but architectural conclusion holds

2. Eval not preprocessed — runs live augmentation (slow, ~20-30 min, once only)

3. num_workers=0 — Windows multiprocessing deadlock fix

4. Temporal Attention Gate was FIXED — was applied to single token (no-op),
   now correctly applied to 32-node GAT sequence before pooling

---

## TENSORBOARD

    conda activate deepfake-dl
    tensorboard --logdir "D:\results"
    Open: http://localhost:6006

---

## NEXT STEPS WHEN BASELINE FINISHES

1. Note final EER from terminal output
2. Run full model command above
3. When full model finishes, run evaluate.py on both checkpoints
4. Compare per-codec EER table
5. Run with --tsne flag for embedding visualisation
