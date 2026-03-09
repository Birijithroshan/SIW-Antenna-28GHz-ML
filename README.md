# CIRCULAR-SIW-Antenna-ML

> **Machine Learning-Based Design Prediction for a Triple-Band Circular Substrate Integrated Waveguide (SIW) Antenna at 28 GHz**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Antenna Design](#antenna-design)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

This project applies supervised machine learning to **predict the resonant frequencies and bandwidths** of a triple-band circular SIW antenna designed for **28 GHz 5G mmWave and satellite communications** (K/Ka-band). Instead of running time-consuming full-wave EM simulations (e.g., CST, HFSS) for every new design, a trained ML model provides near-instant predictions from antenna geometry parameters.

**Key highlights:**
- 1000-sample physics-based dataset
- 10 geometric input parameters → 6 RF output predictions
- 7 ML models benchmarked; best model auto-saved
- Full metrics: MAE, MSE, RMSE, R², Variance Score
- Interactive & batch prediction tools

---

## Antenna Design

The antenna is a **circular SIW structure** with four concentric ring slots etched on the top conductor. It operates simultaneously in three bands within the K/Ka frequency range (~28–36 GHz), making it suitable for:

- 5G NR mmWave (n257/n258/n261 bands)
- Ka-band satellite communications
- Dual-polarization radar systems

### Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `S1_mm` | Ring slot radius 1 (mm) | 5.50 – 6.50 |
| `S2_mm` | Ring slot radius 2 (mm) | 4.20 – 5.70 |
| `S3_mm` | Ring slot radius 3 (mm) | 3.00 – 4.70 |
| `S4_mm` | Ring slot radius 4 (mm) | 1.80 – 3.70 |
| `d_mm` | Via diameter (mm) | 0.45 – 0.65 |
| `Wf_mm` | Feed line width (mm) | 1.00 – 1.20 |
| `Lf_mm` | Feed line length (mm) | 5.00 – 10.00 |
| `RSIW_mm` | SIW cavity radius (mm) | 7.00 – 9.00 |
| `p_mm` | Via pitch / spacing (mm) | 0.80 – 1.20 |
| `h_mm` | Substrate thickness (mm) | 0.508 – 0.787 |

> **Physical constraint:** S1 > S2 > S3 > S4 (concentric ring ordering)

### Output Parameters

| Parameter | Description | Typical Band |
|-----------|-------------|--------------|
| `f1_GHz` | Resonant frequency — Band 1 (GHz) | ~28 GHz (K-band) |
| `f2_GHz` | Resonant frequency — Band 2 (GHz) | ~32 GHz (Ka-band lower) |
| `f3_GHz` | Resonant frequency — Band 3 (GHz) | ~34 GHz (Ka-band upper) |
| `BW1_GHz` | -10 dB Bandwidth at f1 (GHz) | — |
| `BW2_GHz` | -10 dB Bandwidth at f2 (GHz) | — |
| `BW3_GHz` | -10 dB Bandwidth at f3 (GHz) | — |

---

## Dataset

- **File:** `dataset_1000samples.csv`
- **Samples:** 1000
- **Generation:** `generate_dataset.py` — physics-based parametric sampling with enforced geometric constraints (S1 > S2 > S3 > S4)
- **Split:** 80% training / 20% testing

---

## Machine Learning Models

Seven models are trained and compared. The best-performing model (by mean R² across all 6 outputs) is automatically saved as `best_model.pkl`.

| # | Model | Type |
|---|-------|------|
| 1 | Linear Regression | Baseline linear |
| 2 | Decision Tree | CART recursive splitting |
| 3 | Random Forest | Bagging ensemble (200 trees) |
| 4 | Lasso | L1-regularised linear |
| 5 | **ExtraTrees** | Extremely Randomised Trees *(typically best)* |
| 6 | CatBoost | Gradient boosting with symmetric trees |
| 7 | Stacking Ensemble | Averaging of top-3 models |

All models use `MultiOutputRegressor` wrapping where needed to handle 6 simultaneous outputs.

---

## Project Structure

```
CIRCULAR-SIW-Antenna-ML/
│
├── dataset_1000samples.csv     # 1000-sample generated dataset
├── generate_dataset.py         # Dataset generation (physics-based sampling)
├── train_models.py             # Train all 7 models, evaluate, save best
├── predict_antenna.py          # Interactive single-design prediction tool
├── batch_test.py               # Batch prediction for multiple designs
├── generate_plots.py           # Publication-quality result plots
│
├── training_results.csv        # Per-model metrics after training
├── best_model.pkl              # Saved best ML model (generated after training)
├── scaler.pkl                  # Fitted StandardScaler (generated after training)
├── feature_names.pkl           # Input feature names list
│
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8 or higher

```bash
# 1. Clone the repository
git clone https://github.com/Birijithroshan/CIRCULAR-SIW-Antenna-ML.git
cd CIRCULAR-SIW-Antenna-ML

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Generate Dataset *(optional, dataset already included)*
```bash
python generate_dataset.py
```

### Step 2 — Train Models
```bash
python train_models.py
```
Outputs:
- `best_model.pkl` — saved best model
- `scaler.pkl` — fitted feature scaler
- `training_results.csv` — full metrics table
- Several comparison plot `.png` files

### Step 3a — Interactive Prediction
```bash
python predict_antenna.py
```
Enter 10 antenna geometry values when prompted and get instant frequency + bandwidth predictions.

### Step 3b — Batch Prediction
```bash
python batch_test.py
```
Runs predictions on a set of predefined designs and prints a comparison table.

### Step 4 — Generate Plots *(optional)*
```bash
python generate_plots.py
```
Produces publication-ready figures for model comparison, feature importance, true vs predicted scatter plots, and more.

---

## Results

After training, metrics are printed to the console and saved to `training_results.csv`. Example outputs include:

- **Per-output R² scores** for all 6 frequency/bandwidth targets
- **MAE / RMSE** for each output
- **Feature importance** rankings
- **True vs Predicted** scatter plots

> ExtraTrees and Stacking Ensemble typically achieve **R² > 0.97** across all outputs.

---

## Requirements

| Library | Version |
|---------|---------|
| pandas | ≥ 1.5.0 |
| numpy | ≥ 1.23.0 |
| scikit-learn | ≥ 1.2.0 |
| matplotlib | ≥ 3.6.0 |
| seaborn | ≥ 0.12.0 |
| xgboost | ≥ 1.7.0 |
| lightgbm | ≥ 3.3.0 |
| catboost | ≥ 1.1.0 |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Developed by [Birijithroshan](https://github.com/Birijithroshan)*
