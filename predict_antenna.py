"""
Interactive prediction script for 28 GHz Triple-Band Circular SIW Antenna
Loads trained model and scaler, prompts for 6 inputs, predicts 6 outputs.
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
    print("   Model and scaler loaded successfully.\n")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please run train_models.py first.")
    raise SystemExit(1)

# ========================================
# INPUT PARAMETER RANGES
# ========================================
PARAM_INFO = {
    'S1':    ('Ring slot radius 1 (mm)',     5.50,  6.50),
    'S2':    ('Ring slot radius 2 (mm)',     4.20,  5.70),
    'S3':    ('Ring slot radius 3 (mm)',     3.00,  4.70),
    'S4':    ('Ring slot radius 4 (mm)',     1.80,  3.70),
    'd_via': ('Via diameter (mm)',           0.45,  0.65),
    'Wf':    ('Feed line width (mm)',        1.00,  1.20),
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
if not (values['S1'] > values['S2'] > values['S3'] > values['S4']):
    print("\n  WARNING: Physical constraint S1 > S2 > S3 > S4 is not satisfied.")
    print("  Results may be less accurate.")

# ========================================
# FEATURE ENGINEERING (same as training)
# ========================================
X = pd.DataFrame([values])
X['S1_S2_product'] = X['S1'] * X['S2']
X['S3_S4_product'] = X['S3'] * X['S4']
X['dvia_Wf_product'] = X['d_via'] * X['Wf']
X['slot_sum'] = X['S1'] + X['S2'] + X['S3'] + X['S4']
X['slot_range'] = X['S1'] - X['S4']

X_sc = scaler.transform(X)

# ========================================
# PREDICT
# ========================================
pred = model.predict(X_sc)[0]
output_cols = ['Freq1_GHz', 'Freq2_GHz', 'Freq3_GHz',
               'Bandwidth1_MHz', 'Bandwidth2_MHz', 'Bandwidth3_MHz']
results = dict(zip(output_cols, pred))

# ========================================
# DISPLAY RESULTS
# ========================================
print("\n" + "=" * 70)
print("   PREDICTION RESULTS")
print("=" * 70)
print(f"\n  {'Resonant Frequencies':}")
print(f"    Frequency 1: {results['Freq1_GHz']:.3f} GHz  (K-band ~28 GHz)")
print(f"    Frequency 2: {results['Freq2_GHz']:.3f} GHz  (Ka-band lower ~32 GHz)")
print(f"    Frequency 3: {results['Freq3_GHz']:.3f} GHz  (Ka-band upper ~34 GHz)")
print(f"\n  {'Bandwidths':}")
print(f"    Bandwidth 1: {results['Bandwidth1_MHz']:.1f} MHz")
print(f"    Bandwidth 2: {results['Bandwidth2_MHz']:.1f} MHz")
print(f"    Bandwidth 3: {results['Bandwidth3_MHz']:.1f} MHz")

# Performance analysis
print("\n" + "-" * 70)
print("  Performance Analysis:")
if 27.0 <= results['Freq1_GHz'] <= 28.5:
    print("  [OK] Freq1 is in the 28 GHz 5G NR band (n257/n258).")
else:
    print(f"  [--] Freq1 ({results['Freq1_GHz']:.3f} GHz) is outside the 28 GHz band.")

total_bw = (results['Bandwidth1_MHz'] + results['Bandwidth2_MHz']
            + results['Bandwidth3_MHz'])
print(f"  [OK] Total bandwidth across 3 bands: {total_bw:.0f} MHz")

# Application suitability
print("\n  Application Suitability:")
apps = []
if 26.5 <= results['Freq1_GHz'] <= 29.5:
    apps.append("5G mmWave (28 GHz)")
if 30.0 <= results['Freq2_GHz'] <= 33.0:
    apps.append("Ka-band satellite")
if 33.0 <= results['Freq3_GHz'] <= 36.0:
    apps.append("Ka-band radar / VSAT")
if apps:
    for app in apps:
        print(f"    - {app}")
else:
    print("    - Custom multi-band application")

print("\n" + "=" * 70)
