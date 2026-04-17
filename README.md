# Applied DL Project — Deepfake Audio Detection

## Group Members
- Rohith Bharathi
- Gaurav Bhatnagar
- Naveen Nagaraja Sudhakar

## Git Repo
https://github.com/roh1thbharathi/applied-dl-project-2

---

## Project Area
Perception — Audio (.wav, .flac)

---

## Problem Statement (HW2)
In audio perception, modern deepfake audio detectors are trained on clean, uncompressed speech but deployed in environments where audio has been codec-compressed (e.g., MP3, AAC, Opus). It is unclear how robust these detectors are to real-world codec degradation, and whether newer generative models are more or less affected than older ones.

---

## Datasets (HW2)
- **ASVspoof 2019 / 2021** — standard benchmark dataset for spoofed and deepfake audio detection
- **WaveFake** — synthetic speech from multiple modern vocoders (WaveGlow, MelGAN, HiFiGAN, etc.)
- Codec compression variants applied using `ffmpeg` and `torchaudio` to simulate real-world degradation

---

## Proposed NN Contribution
We modify an existing deepfake audio detector using a **Domain Adversarial Neural Network (DANN)**-style architecture:

### Architecture
1. **Shared Encoder** — extracts features from audio
2. **Head 1: Deepfake Detector** — classifies real vs fake (minimize loss ✓)
3. **Gradient Reversal Layer (GRL)** — flips the training signal
4. **Head 2: Codec Classifier** — tries to identify codec type (maximize loss ✗)

### How it works
The GRL forces the encoder to learn features that are useful for detecting deepfakes **but useless for identifying which codec was used**. The result is a codec-invariant encoder — robustness comes from the architecture itself, not pre/post processing.

---

## Goals

### Quantitative
- Equal Error Rate (EER) and AUC across codec types and bitrates
- Measure performance degradation vs uncompressed baseline

### Qualitative
- Identify which generator generations are most vulnerable to codec-induced misclassification
- Visualize spectral and feature-level differences pre/post compression

---

## Use Case
Real-time deepfake detection in telecom and voice authentication pipelines (phone calls, VoIP, WhatsApp) where audio is always codec-compressed before reaching the detector.

---

## Project Structure
```
applied-dl-project-2/
├── data/
│   ├── raw/          # Original datasets (ASVspoof, WaveFake)
│   └── processed/    # Codec-compressed variants
├── src/              # Source code and scripts
├── notebooks/        # EDA and experimentation
├── models/           # Trained model checkpoints
├── docs/             # Documentation and write-ups
├── results/          # Plots, metrics, outputs
├── requirements.txt
└── .gitignore
```

---

## Dependencies
```
torch
torchaudio
librosa
numpy
pandas
scikit-learn
matplotlib
shap
```

---

## HW Submissions
- **HW1 (due 3/23):** Group formed, repo created, email sent ✅
- **HW2 (due 3/30):** Problem, datasets, goals submitted ✅
