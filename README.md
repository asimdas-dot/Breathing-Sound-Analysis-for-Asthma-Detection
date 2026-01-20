# Breathing-Sound-Analysis-for-Asthma-Detection
Asthma and COPD pose significant challenges to human health and global healthcare systems.The aim is to develop an AI-based diagnostic system capable of accurately distinguishing these conditions, thereby enhancing early detection and clinical management. presents the first AI system that enhance the diagnostic ACC of asthma.
# Asthma Detection using Respiratory & Cough Sounds

## Overview
[cite_start]This project implements an automated AI system to distinguish between asthma, COPD, and healthy subjects using audio signals[cite: 8].

## Dataset Specifications
- [cite_start]**Cough Sounds**: 48,000 Hz sampling rate[cite: 72].
- [cite_start]**Respiratory Sounds**: 4,000 Hz sampling rate[cite: 73].
- [cite_start]**Processing**: Non-cough segments and noise removed; signals standardized to equal-length segments[cite: 66, 67].

## Methodology
1. [cite_start]**Signal Transformation**: STFT, Gabor, or CWT to generate 2D spectrograms[cite: 97, 98].
2. [cite_start]**Feature Extraction**: Pre-trained lightweight CNNs (ShuffleNet, SqueezeNet, MobileNet, EfficientNet)[cite: 529].
3. [cite_start]**Feature Selection**: Relief or Neighborhood Component Analysis (NCA) to reduce dimensionality[cite: 15, 153].
4. [cite_start]**Classification**: Ensemble of SVM, NN, RF, KNN, and DT using majority voting[cite: 11, 14].

## Requirements
- Python 3.8+
- librosa, scipy (Signal processing)
- PyTorch/Torchvision (Deep Learning)
- scikit-learn (Machine Learning & Feature Selection)