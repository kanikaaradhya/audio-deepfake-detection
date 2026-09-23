# Audio DeepFake Detection

**An ML pipeline that classifies audio as real human speech or AI-generated using an XGBoost + LightGBM ensemble over handcrafted acoustic features — achieving 99.82% accuracy on the Fake-or-Real dataset, served as a FastAPI REST endpoint.**

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/-XGBoost-EC6C1E?style=flat-square) ![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat-square) ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Librosa](https://img.shields.io/badge/-Librosa-8B4513?style=flat-square)

> **Note on this repository:** This is a technical write-up of the system's design, architecture, and implementation — not the original source tree. It documents the approach, decisions, and trade-offs made while building the project.

---

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.82%** |
| **F1 Score** | **0.9982** |
| Precision (Real) | 1.00 |
| Recall (Real) | 1.00 |
| Precision (Fake) | 1.00 |
| Recall (Fake) | 1.00 |
| Training samples | 11,164 |
| Test samples | 2,792 |
| Feature vector size | 121 dimensions |
| Dataset | Fake-or-Real (for-2sec), York University |

## Overview

As deepfake audio becomes more convincing (modern TTS systems like WaveNet, DeepVoice 3, and Amazon Polly produce near-human speech), the ability to detect synthetic audio programmatically becomes critical — for media verification, fraud prevention, and forensic analysis.

This project takes a feature-engineering approach rather than an end-to-end deep learning one: extract meaningful acoustic features (MFCCs, spectral centroid, chroma, etc.) from raw audio, then classify with an ensemble of gradient-boosted trees. The result is a lightweight, interpretable pipeline that runs on CPU and achieves near-perfect accuracy on the standard FoR benchmark.

Built as a seminar project (22IS4SRINT) at BMS College of Engineering, 2022–23, then extended with the classifier and API.

## My Role
- Designed and implemented the feature extraction pipeline (8 feature types → 121-dimensional vector)
- Built and trained the XGBoost + LightGBM soft-voting ensemble
- Wrapped the trained model as a FastAPI REST endpoint for inference
- Ran the initial audio analysis (waveform, MFCC, spectrogram, spectral centroid, chroma comparisons between real and fake audio) that informed which features to extract

## Architecture

```
                        Audio File (.wav)
                              │
                              ▼
               ┌──────────────────────────┐
               │    Feature Extraction     │
               │       (librosa)           │
               │                           │
               │  • MFCC (40 coefficients) │
               │  • Spectral Centroid      │
               │  • Spectral Bandwidth     │
               │  • Spectral Rolloff       │
               │  • Zero Crossing Rate     │
               │  • Chroma (12 bins)       │
               │  • RMS Energy             │
               │  • Spectral Contrast      │
               │         (7 bands)         │
               └────────────┬─────────────┘
                            │
                     121-dim feature vector
                            │
                            ▼
               ┌──────────────────────────┐
               │    Ensemble Classifier    │
               │                           │
               │  ┌───────┐  ┌──────────┐ │
               │  │XGBoost│  │ LightGBM │ │
               │  └───┬───┘  └────┬─────┘ │
               │      │           │        │
               │      └─────┬─────┘        │
               │     soft voting           │
               │   (avg probabilities)     │
               └────────────┬──────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │   FastAPI REST Endpoint   │
               │                           │
               │  POST /predict            │
               │  ← audio file             │
               │  → { label, confidence }  │
               └──────────────────────────┘
```

## Key Design Decisions

### Why feature engineering over deep learning?
A CNN or transformer trained on raw spectrograms might achieve similar accuracy, but requires a GPU, a much larger dataset to avoid overfitting, and is a black box. The feature-engineering approach produces a 121-dimensional vector per clip, trains in under a minute on CPU, and every feature (spectral centroid, MFCCs, chroma) has a known acoustic interpretation — you can explain *why* the model flagged a clip as fake. For a detection tool where explainability matters (forensics, media verification), that's a real advantage.

### Why XGBoost + LightGBM ensemble?
Both are gradient-boosted tree models, but they split differently: XGBoost uses a level-wise strategy; LightGBM uses leaf-wise growth. Ensembling them via soft voting (averaging predicted probabilities) captures patterns that either model alone might miss. On this dataset the individual models each hit ~99.7%, and the ensemble pushes to 99.82%.

### Why the FoR for-2sec variant?
The for-2sec variant truncates all clips to a fixed 2-second duration, which eliminates length as a confounding feature. Without this, the model could learn to classify based on clip duration rather than acoustic content (real speech samples in FoR-original average 5 seconds; synthetic ones average 2.3 seconds). Fixed-length clips force the model to learn actual acoustic differences.

### Why mean + std aggregation for time-series features?
MFCCs, chroma, and spectral features produce a value per time frame — for a 2-second clip at 22,050 Hz, that's ~87 frames. Feeding raw frame-level features would create a variable-length input. Taking the mean and standard deviation across time compresses each feature into two stable numbers while preserving both the central tendency and the variability — which turns out to be highly discriminative for real vs. synthetic speech.

## Deep-Dive Documentation

| Document | Covers |
|----------|--------|
| **[Feature Engineering](docs/feature-engineering.md)** | All 8 feature types, why each matters for deepfake detection, extraction code |
| **[Model Training](docs/model-training.md)** | Dataset prep, XGBoost + LightGBM config, ensemble strategy, evaluation |
| **[API Endpoint](docs/api.md)** | FastAPI `/predict` route, request/response format, deployment |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Feature Extraction | librosa, NumPy |
| Models | XGBoost, LightGBM, Scikit-learn (VotingClassifier) |
| API | FastAPI, Uvicorn |
| Dataset | Fake-or-Real for-2sec (York University) — 13,956 clips |
| Serialisation | joblib |

Built for the Seminar – Internship Involving Social Activity course (22IS4SRINT), Dept. of ISE, BMS College of Engineering, 2022–23. Extended with classifier and API.

## References

1. Reimao, R. & Tzerpos, V. — *FoR: Fake or Real Dataset for Synthetic Speech Detection*, York University
2. Iqbal, F. et al. — *Deepfake Audio Detection via Feature Engineering and Machine Learning* (CEUR-WS Vol. 3318)
3. Pianese, A. et al. — *Deepfake Audio Detection by Speaker Verification* (arXiv:2209.14098)
