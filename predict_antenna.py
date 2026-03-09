"""
Interactive prediction script for 28 GHz Triple-Band Circular SIW Antenna
Loads trained model and scaler, prompts for 10 inputs, predicts 6 outputs.
"""

import pickle
import numpy as np
import pandas as pd

print("=" * 70)
print("   28 GHz TRIPLE-BAND CIRCULAR SIW ANTENNA - PREDICTION TOOL")
print("=" * 70)

# ========================================
# LOAD MODEL & SCALER
# ========================================
print("\nLoading model and scaler...")
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    # best_model.pkl holds whichever of the 7 trained models scored highest R²:
    #   LinearRegression  — Ordinary Least Squares (linear)
    #   DecisionTree      — CART recursive binary splitting
    #   RandomForest      — Bagging ensemble of 200 CART trees
    #   Lasso             — L1-regularised Linear Regression
    #   ExtraTrees        — Extremely Randomised Trees (fully random splits)
    #   CatBoost          — Gradient Boosting with ordered boosting & symmetric trees
    #   Stacking_Ensemble — Averaging ensemble of top-3 models
    print(f"   Loaded model : {type(model).__name__}")
    print("   Scaler and feature names loaded successfully.\n")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please run train_models.py first.")
    raise SystemExit(1)

# ========================================
# INPUT PARAMETER RANGES
# ========================================
PARAM_INFO = {
    'S1_mm':   ('Ring slot radius 1 (mm)',     5.50,  6.50),
    'S2_mm':   ('Ring slot radius 2 (mm)',     4.20,  5.70),
    'S3_mm':   ('Ring slot radius 3 (mm)',     3.00,  4.70),
    'S4_mm':   ('Ring slot radius 4 (mm)',     1.80,  3.70),
    'd_mm':    ('Via diameter (mm)',           0.45,  0.65),
    'Wf_mm':   ('Feed line width (mm)',        1.00,  1.20),
    'Lf_mm':   ('Feed line length (mm)',       5.00, 10.00),
    'RSIW_mm': ('SIW cavity radius (mm)',      7.00,  9.50),
    'p_mm':    ('Via pitch / spacing (mm)',    0.80,  1.40),
    'h_mm':    ('Substrate height (mm)',       0.508, 1.575),
}

print("Input Parameter Ranges:")
print(f"  {'Parameter':<10} {'Description':<30} {'Min':>6} {'Max':>6}")
print("  " + "-" * 56)
for param, (desc, lo, hi) in PARAM_INFO.items():
    print(f"  {param:<10} {desc:<30} {lo:>6.2f} {hi:>6.2f}")
print()

# ========================================
# PROMPT FOR INPUTS
# ========================================
def prompt_float(name, desc, lo, hi):
    while True:
        try:
            val = float(input(f"  Enter {name} - {desc} [{lo:.2f} - {hi:.2f}]: "))
        except ValueError:
            print("    Invalid input. Please enter a number.")
            continue
        if not (lo <= val <= hi):
            print(f"    Value {val} is outside the range [{lo:.2f}, {hi:.2f}]. "
                  "Please try again.")
            continue
        return val

print("Enter antenna parameters:")
values = {}
for param, (desc, lo, hi) in PARAM_INFO.items():
    values[param] = prompt_float(param, desc, lo, hi)

# Physical constraint check
if not (values['S1_mm'] > values['S2_mm'] > values['S3_mm'] > values['S4_mm']):
    print("\n  WARNING: Physical constraint S1 > S2 > S3 > S4 is not satisfied.")
    print("  Results may be less accurate.")

# ========================================
# FEATURE ENGINEERING (same as training)
# ========================================
X = pd.DataFrame([values])
X['S1_S2_product']  = X['S1_mm']   * X['S2_mm']
X['S3_S4_product']  = X['S3_mm']   * X['S4_mm']
X['d_Wf_product']   = X['d_mm']    * X['Wf_mm']
X['RSIW_h_product'] = X['RSIW_mm'] * X['h_mm']
X['slot_sum']       = X['S1_mm'] + X['S2_mm'] + X['S3_mm'] + X['S4_mm']
X['slot_range']     = X['S1_mm'] - X['S4_mm']
X['Lf_p_ratio']     = X['Lf_mm']   / (X['p_mm'] + 1e-9)
# Ensure column order matches training exactly
X = X[feature_names]

X_sc = scaler.transform(X)

# ========================================
# PREDICT
# ========================================
pred = model.predict(X_sc)[0]
output_cols = ['f1_GHz', 'f2_GHz', 'f3_GHz',
               'BW1_GHz', 'BW2_GHz', 'BW3_GHz']
results = dict(zip(output_cols, pred))

# ========================================
# DISPLAY RESULTS
# ========================================
print("\n" + "=" * 70)
print("   PREDICTION RESULTS")
print("=" * 70)
print(f"\n  {'Resonant Frequencies':}")
print(f"    Frequency 1: {results['f1_GHz']:.3f} GHz  (K-band ~28 GHz)")
print(f"    Frequency 2: {results['f2_GHz']:.3f} GHz  (Ka-band lower ~32 GHz)")
print(f"    Frequency 3: {results['f3_GHz']:.3f} GHz  (Ka-band upper ~34 GHz)")
print(f"\n  {'Bandwidths':}")
print(f"    BW1: {results['BW1_GHz']:.3f} GHz")
print(f"    BW2: {results['BW2_GHz']:.3f} GHz")
print(f"    BW3: {results['BW3_GHz']:.3f} GHz")

# Performance analysis
print("\n" + "-" * 70)
print("  Performance Analysis:")
if 27.0 <= results['f1_GHz'] <= 28.5:
    print("  [OK] f1 is in the 28 GHz 5G NR band (n257/n258).")
else:
    print(f"  [--] f1 ({results['f1_GHz']:.3f} GHz) is outside the 28 GHz band.")

total_bw = results['BW1_GHz'] + results['BW2_GHz'] + results['BW3_GHz']
print(f"  [OK] Total bandwidth across 3 bands: {total_bw:.3f} GHz")

# Application suitability
print("\n  Application Suitability:")
apps = []
if 26.5 <= results['f1_GHz'] <= 29.5:
    apps.append("5G mmWave (28 GHz)")
if 30.0 <= results['f2_GHz'] <= 33.0:
    apps.append("Ka-band satellite")
if 33.0 <= results['f3_GHz'] <= 36.0:
    apps.append("Ka-band radar / VSAT")
if apps:
    for app in apps:
        print(f"    - {app}")
else:
    print("    - Custom multi-band application")

print("\n" + "=" * 70)
