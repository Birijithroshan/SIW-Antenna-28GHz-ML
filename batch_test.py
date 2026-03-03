"""
Batch testing script for 28 GHz Triple-Band Circular SIW Antenna.
Tests 5 predefined designs and displays comparison table.
"""

import pickle
import numpy as np
import pandas as pd

print("=" * 90)
print("   28 GHz TRIPLE-BAND CIRCULAR SIW ANTENNA - BATCH TEST")
print("=" * 90)

# ========================================
# LOAD MODEL & SCALER
# ========================================
print("\nLoading model and scaler...")
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("   Loaded successfully.\n")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please run train_models.py first.")
    raise SystemExit(1)

# ========================================
# PREDEFINED TEST DESIGNS
# ========================================
# Columns: S1, S2, S3, S4, d_via, Wf
designs = {
    'Balanced':         {'S1': 6.00, 'S2': 5.00, 'S3': 4.00, 'S4': 3.00,
                         'd_via': 0.55, 'Wf': 1.10},
    'Minimum':          {'S1': 5.50, 'S2': 4.50, 'S3': 3.50, 'S4': 2.50,
                         'd_via': 0.45, 'Wf': 1.00},
    'Maximum':          {'S1': 6.50, 'S2': 5.50, 'S3': 4.50, 'S4': 3.50,
                         'd_via': 0.65, 'Wf': 1.20},
    '28GHz Optimized':  {'S1': 5.80, 'S2': 4.90, 'S3': 3.90, 'S4': 2.90,
                         'd_via': 0.52, 'Wf': 1.08},
    'Wide Bandwidth':   {'S1': 6.30, 'S2': 5.20, 'S3': 4.10, 'S4': 3.10,
                         'd_via': 0.60, 'Wf': 1.18},
}

input_cols = ['S1', 'S2', 'S3', 'S4', 'd_via', 'Wf']
output_cols = ['Freq1_GHz', 'Freq2_GHz', 'Freq3_GHz',
               'Bandwidth1_MHz', 'Bandwidth2_MHz', 'Bandwidth3_MHz']

# ========================================
# FEATURE ENGINEERING (same as training)
# ========================================
X = pd.DataFrame(designs).T[input_cols]
X['S1_S2_product'] = X['S1'] * X['S2']
X['S3_S4_product'] = X['S3'] * X['S4']
X['dvia_Wf_product'] = X['d_via'] * X['Wf']
X['slot_sum'] = X['S1'] + X['S2'] + X['S3'] + X['S4']
X['slot_range'] = X['S1'] - X['S4']

X_sc = scaler.transform(X)

# ========================================
# PREDICT
# ========================================
preds = model.predict(X_sc)
pred_df = pd.DataFrame(preds, index=list(designs.keys()), columns=output_cols)

# ========================================
# DISPLAY RESULTS
# ========================================
print("Input Parameters:")
input_df = X[input_cols]
print(input_df.to_string())

print("\n" + "=" * 90)
print("Prediction Results:")
print(pred_df.round(3).to_string())

print("\n" + "=" * 90)
print("Combined Comparison Table:")
combined = pd.concat([input_df, pred_df.round(3)], axis=1)
print(combined.to_string())

print("\n" + "=" * 90)
print("Summary:")
for design in designs:
    row = pred_df.loc[design]
    total_bw = row['Bandwidth1_MHz'] + row['Bandwidth2_MHz'] + row['Bandwidth3_MHz']
    print(f"  {design:<20}  "
          f"Freq1={row['Freq1_GHz']:.3f} GHz  "
          f"Freq2={row['Freq2_GHz']:.3f} GHz  "
          f"Freq3={row['Freq3_GHz']:.3f} GHz  "
          f"Total BW={total_bw:.0f} MHz")

print("\n" + "=" * 90)
print("   BATCH TEST COMPLETE!")
print("=" * 90)
