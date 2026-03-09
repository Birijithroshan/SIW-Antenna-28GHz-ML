"""
Batch testing script for 28 GHz Triple-Band Circular SIW Antenna.
Tests 5 predefined designs and displays comparison table.
10 Inputs : S1_mm, S2_mm, S3_mm, S4_mm, d_mm, Wf_mm, Lf_mm, RSIW_mm, p_mm, h_mm
 6 Outputs: f1_GHz, f2_GHz, f3_GHz, BW1_GHz, BW2_GHz, BW3_GHz
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
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("   Loaded successfully.\n")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please run train_models.py first.")
    raise SystemExit(1)

# ========================================
# PREDEFINED TEST DESIGNS
# ========================================
# Columns: S1_mm, S2_mm, S3_mm, S4_mm, d_mm, Wf_mm, Lf_mm, RSIW_mm, p_mm, h_mm
designs = {
    'Balanced':        {'S1_mm': 6.00, 'S2_mm': 5.00, 'S3_mm': 4.00, 'S4_mm': 3.00,
                        'd_mm': 0.55, 'Wf_mm': 1.10, 'Lf_mm': 7.50,
                        'RSIW_mm': 8.25, 'p_mm': 1.10, 'h_mm': 1.00},
    'Minimum':         {'S1_mm': 5.50, 'S2_mm': 4.50, 'S3_mm': 3.50, 'S4_mm': 2.50,
                        'd_mm': 0.45, 'Wf_mm': 1.00, 'Lf_mm': 5.00,
                        'RSIW_mm': 7.00, 'p_mm': 0.80, 'h_mm': 0.508},
    'Maximum':         {'S1_mm': 6.50, 'S2_mm': 5.50, 'S3_mm': 4.50, 'S4_mm': 3.50,
                        'd_mm': 0.65, 'Wf_mm': 1.20, 'Lf_mm': 10.00,
                        'RSIW_mm': 9.50, 'p_mm': 1.40, 'h_mm': 1.575},
    '28GHz Optimized': {'S1_mm': 5.80, 'S2_mm': 4.90, 'S3_mm': 3.90, 'S4_mm': 2.90,
                        'd_mm': 0.52, 'Wf_mm': 1.08, 'Lf_mm': 7.00,
                        'RSIW_mm': 8.00, 'p_mm': 1.05, 'h_mm': 0.762},
    'Wide Bandwidth':  {'S1_mm': 6.30, 'S2_mm': 5.20, 'S3_mm': 4.10, 'S4_mm': 3.10,
                        'd_mm': 0.60, 'Wf_mm': 1.18, 'Lf_mm': 8.50,
                        'RSIW_mm': 8.60, 'p_mm': 1.20, 'h_mm': 1.270},
}

input_cols  = ['S1_mm', 'S2_mm', 'S3_mm', 'S4_mm', 'd_mm',
               'Wf_mm', 'Lf_mm', 'RSIW_mm', 'p_mm', 'h_mm']
output_cols = ['f1_GHz', 'f2_GHz', 'f3_GHz',
               'BW1_GHz', 'BW2_GHz', 'BW3_GHz']

# ========================================
# FEATURE ENGINEERING (same as training)
# ========================================
X = pd.DataFrame(designs).T[input_cols]
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
    total_bw = row['BW1_GHz'] + row['BW2_GHz'] + row['BW3_GHz']
    print(f"  {design:<20}  "
          f"f1={row['f1_GHz']:.3f} GHz  "
          f"f2={row['f2_GHz']:.3f} GHz  "
          f"f3={row['f3_GHz']:.3f} GHz  "
          f"Total BW={total_bw:.3f} GHz")

print("\n" + "=" * 90)
print("   BATCH TEST COMPLETE!")
print("=" * 90)
