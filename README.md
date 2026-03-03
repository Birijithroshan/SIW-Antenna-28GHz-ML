# SIW-Antenna-28GHz-ML

ML-based prediction for 28 GHz Triple-Band Circular Substrate Integrated Waveguide (SIW) Antenna.

## Overview

This project applies machine learning to predict the resonant frequencies and bandwidths of a
triple-band circular SIW antenna operating at 28 GHz (K/Ka-band), targeting 5G mmWave and
satellite communications applications.

### Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| S1 | Ring slot radius 1 (mm) | 5.50 – 6.50 |
| S2 | Ring slot radius 2 (mm) | 4.20 – 5.70 |
| S3 | Ring slot radius 3 (mm) | 3.00 – 4.70 |
| S4 | Ring slot radius 4 (mm) | 1.80 – 3.70 |
| d_via | Via diameter (mm) | 0.45 – 0.65 |
| Wf | Feed line width (mm) | 1.00 – 1.20 |

Physical constraint: **S1 > S2 > S3 > S4**

### Output Parameters

| Parameter | Description | Band |
|-----------|-------------|------|
| Freq1_GHz | Resonant frequency 1 (GHz) | K-band ~28 GHz |
| Freq2_GHz | Resonant frequency 2 (GHz) | Ka-band lower ~32 GHz |
| Freq3_GHz | Resonant frequency 3 (GHz) | Ka-band upper ~34 GHz |
| Bandwidth1_MHz | Bandwidth at Freq1 (MHz) | — |
| Bandwidth2_MHz | Bandwidth at Freq2 (MHz) | — |
| Bandwidth3_MHz | Bandwidth at Freq3 (MHz) | — |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the 1000-sample dataset
python generate_dataset.py

# 3. Train all ML models and save the best one
python train_models.py

# 4a. Interactive single-design prediction
python predict_antenna.py

# 4b. Batch test with 5 predefined designs
python batch_test.py
```

## Model Performance

Training evaluates the following models and saves the best one as `best_model.pkl`:

| Model | Type |
|-------|------|
| ExtraTreesRegressor | Native multi-output |
| RandomForestRegressor | Native multi-output |
| GradientBoostingRegressor | MultiOutputRegressor wrapper |
| XGBRegressor | MultiOutputRegressor wrapper (optional) |
| LGBMRegressor | MultiOutputRegressor wrapper (optional) |
| CatBoostRegressor | MultiOutputRegressor wrapper (optional) |
| StackingRegressor | Ensemble of top 3 models + Ridge |

Results are saved to `training_results.csv` and visualised in `model_comparison.png` and
`bandwidth_predictions.png`.

## Project Structure

```
SIW-Antenna-28GHz-ML/
├── generate_dataset.py      # Generate 1000-sample physics-based dataset
├── train_models.py          # Train ML models and select best
├── predict_antenna.py       # Interactive single-design prediction
├── batch_test.py            # Batch test with 5 predefined designs
├── requirements.txt         # Python dependencies
└── README.md
```

## Reference

> Birijithroshan et al., "Triple-Band Circular SIW Antenna for 28 GHz 5G Applications",
> *IEEE Transactions on Antennas and Propagation* (submitted).
