"""
Generate 1000-sample dataset for 28 GHz Triple-Band Circular SIW Antenna
10 Inputs : S1_mm, S2_mm, S3_mm, S4_mm, d_mm, Wf_mm, Lf_mm, RSIW_mm, p_mm, h_mm
 6 Outputs: f1_GHz, f2_GHz, f3_GHz, BW1_GHz, BW2_GHz, BW3_GHz
"""

import numpy as np
import pandas as pd
import os

print("=" * 80)
print("      GENERATING 1000-SAMPLE DATASET - TRIPLE-BAND SIW ANTENNA")
print("=" * 80)

np.random.seed(42)
n_samples = 1000

print(f"\n[1/5] Generating {n_samples} input parameter sets...")

# Ring slot radii (physical constraint: S1 > S2 > S3 > S4)
S1 = np.random.uniform(5.50, 6.50, n_samples)
S2_offset = np.random.uniform(0.80, 1.20, n_samples)
S3_offset = np.random.uniform(0.80, 1.20, n_samples)
S4_offset = np.random.uniform(0.80, 1.20, n_samples)

S2 = S1 - S2_offset
S3 = S2 - S3_offset
S4 = S3 - S4_offset

# Enforce minimum gaps
S2 = np.clip(S2, 4.20, S1 - 0.30)
S3 = np.clip(S3, 3.00, S2 - 0.30)
S4 = np.clip(S4, 1.80, S3 - 0.30)

# Via diameter (d_mm)
d = np.random.uniform(0.45, 0.65, n_samples)

# Feed line width (Wf_mm)
Wf = np.random.uniform(1.00, 1.20, n_samples)

# Feed line length (Lf_mm)
Lf = np.random.uniform(5.00, 10.00, n_samples)

# SIW cavity radius (RSIW_mm) — dominant factor for resonant frequency
RSIW = np.random.uniform(7.00, 9.50, n_samples)

# Via pitch / spacing (p_mm)
p = np.random.uniform(0.80, 1.40, n_samples)

# Substrate height (h_mm)
h = np.random.uniform(0.508, 1.575, n_samples)

print(f"   S1_mm:   {S1.min():.3f} – {S1.max():.3f} mm")
print(f"   S2_mm:   {S2.min():.3f} – {S2.max():.3f} mm")
print(f"   S3_mm:   {S3.min():.3f} – {S3.max():.3f} mm")
print(f"   S4_mm:   {S4.min():.3f} – {S4.max():.3f} mm")
print(f"   d_mm:    {d.min():.3f} – {d.max():.3f} mm")
print(f"   Wf_mm:   {Wf.min():.3f} – {Wf.max():.3f} mm")
print(f"   Lf_mm:   {Lf.min():.3f} – {Lf.max():.3f} mm")
print(f"   RSIW_mm: {RSIW.min():.3f} – {RSIW.max():.3f} mm")
print(f"   p_mm:    {p.min():.3f} – {p.max():.3f} mm")
print(f"   h_mm:    {h.min():.3f} – {h.max():.3f} mm")

# Centers used for normalisation in empirical formulae
S1_c, S2_c, S3_c, S4_c = 6.00, 5.00, 4.00, 3.00
d_c, Wf_c, Lf_c = 0.55, 1.10, 7.50
RSIW_c, p_c, h_c = 8.25, 1.10, 1.00

# ========================================
# FREQUENCY CALCULATIONS (Physics-inspired empirical model)
# ========================================
print(f"\n[2/5] Calculating triple-band resonant frequencies...")

# f1 — K-band ~27 GHz
f1 = (27.0
      - 0.55 * (RSIW - RSIW_c)    # larger cavity  → lower resonance
      - 0.30 * (h    - h_c)        # thicker substrate → loading effect
      + 0.20 * (S1   - S1_c)
      + 0.15 * (S2   - S2_c)
      + 0.10 * (S3   - S3_c)
      + 0.08 * (S4   - S4_c)
      + 0.30 * (d    - d_c)
      + 0.15 * (Wf   - Wf_c)
      - 0.02 * (Lf   - Lf_c)
      - 0.10 * (p    - p_c)
      + 0.05 * (S1   - S1_c) * (S2 - S2_c)
      + np.random.normal(0, 0.015, n_samples))
f1_GHz = np.clip(np.round(f1, 3), 25.00, 29.50)

# f2 — Ka-band lower ~32 GHz
f2 = (32.0
      - 0.60 * (RSIW - RSIW_c)
      - 0.28 * (h    - h_c)
      + 0.25 * (S1   - S1_c)
      + 0.20 * (S2   - S2_c)
      + 0.15 * (S3   - S3_c)
      + 0.12 * (S4   - S4_c)
      + 0.35 * (d    - d_c)
      + 0.20 * (Wf   - Wf_c)
      - 0.02 * (Lf   - Lf_c)
      - 0.12 * (p    - p_c)
      + 0.06 * (S1   - S1_c) * (S3 - S3_c)
      + np.random.normal(0, 0.018, n_samples))
f2_GHz = np.clip(np.round(f2, 3), 29.50, 34.50)

# f3 — Ka-band upper ~34 GHz
f3 = (34.0
      - 0.65 * (RSIW - RSIW_c)
      - 0.25 * (h    - h_c)
      + 0.30 * (S1   - S1_c)
      + 0.22 * (S2   - S2_c)
      + 0.18 * (S3   - S3_c)
      + 0.14 * (S4   - S4_c)
      + 0.30 * (d    - d_c)
      + 0.18 * (Wf   - Wf_c)
      - 0.02 * (Lf   - Lf_c)
      - 0.08 * (p    - p_c)
      + 0.07 * (S2   - S2_c) * (S4 - S4_c)
      + np.random.normal(0, 0.020, n_samples))
f3_GHz = np.clip(np.round(f3, 3), 31.50, 37.00)

print(f"   f1: {f1_GHz.min():.3f} – {f1_GHz.max():.3f} GHz  (K-band)")
print(f"   f2: {f2_GHz.min():.3f} – {f2_GHz.max():.3f} GHz  (Ka-band lower)")
print(f"   f3: {f3_GHz.min():.3f} – {f3_GHz.max():.3f} GHz  (Ka-band upper)")

# ========================================
# BANDWIDTH CALCULATIONS (in GHz)
# ========================================
print(f"\n[3/5] Calculating triple-band bandwidths (GHz)...")

# BW1_GHz  (~1.95 GHz)
bw1 = (1.95
       + 0.12 * (f1_GHz - 27.0)
       + 0.20 * (Wf   - Wf_c)
       + 0.08 * (h    - h_c)
       + 0.06 * (S1   - S1_c)
       + 0.04 * (S2   - S2_c)
       + 0.05 * (d    - d_c)
       + 0.03 * (Lf   - Lf_c) * 0.01
       + np.random.normal(0, 0.010, n_samples))
BW1_GHz = np.clip(np.round(bw1, 3), 1.80, 2.15)

# BW2_GHz  (~2.55 GHz)
bw2 = (2.55
       + 0.10 * (f2_GHz - 32.0)
       + 0.18 * (Wf   - Wf_c)
       + 0.07 * (h    - h_c)
       + 0.05 * (S1   - S1_c)
       + 0.04 * (S3   - S3_c)
       + 0.04 * (d    - d_c)
       + 0.03 * (Lf   - Lf_c) * 0.01
       + np.random.normal(0, 0.012, n_samples))
BW2_GHz = np.clip(np.round(bw2, 3), 2.40, 2.75)

# BW3_GHz  (~2.10 GHz)
bw3 = (2.10
       + 0.11 * (f3_GHz - 34.0)
       + 0.19 * (Wf   - Wf_c)
       + 0.06 * (h    - h_c)
       + 0.05 * (S2   - S2_c)
       + 0.04 * (S4   - S4_c)
       + 0.04 * (d    - d_c)
       + 0.03 * (Lf   - Lf_c) * 0.01
       + np.random.normal(0, 0.011, n_samples))
BW3_GHz = np.clip(np.round(bw3, 3), 1.95, 2.35)

print(f"   BW1: {BW1_GHz.min():.3f} – {BW1_GHz.max():.3f} GHz")
print(f"   BW2: {BW2_GHz.min():.3f} – {BW2_GHz.max():.3f} GHz")
print(f"   BW3: {BW3_GHz.min():.3f} – {BW3_GHz.max():.3f} GHz")

# ========================================
# CREATE DATAFRAME
# ========================================
print(f"\n[4/5] Building DataFrame...")


df = pd.DataFrame({
    # Inputs (10)
    'S1_mm':   np.round(S1,   3),
    'S2_mm':   np.round(S2,   3),
    'S3_mm':   np.round(S3,   3),
    'S4_mm':   np.round(S4,   3),
    'd_mm':    np.round(d,    3),
    'Wf_mm':   np.round(Wf,   3),
    'Lf_mm':   np.round(Lf,   3),
    'RSIW_mm': np.round(RSIW, 3),
    'p_mm':    np.round(p,    3),
    'h_mm':    np.round(h,    3),
    # Outputs (6)
    'f1_GHz':  f1_GHz,
    'f2_GHz':  f2_GHz,
    'f3_GHz':  f3_GHz,
    'BW1_GHz': BW1_GHz,
    'BW2_GHz': BW2_GHz,
    'BW3_GHz': BW3_GHz,
})

# ========================================
# SAVE
# ========================================
print(f"\n[5/5] Saving dataset...")

output_file = 'dataset_1000samples.csv'
df.to_csv(output_file, index=False)
file_size = os.path.getsize(output_file) / 1024

print(f"\n{'=' * 80}")
print(f"   DATASET SAVED SUCCESSFULLY!")
print(f"{'=' * 80}")
print(f"   File    : {output_file}")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {len(df.columns)}  (10 inputs + 6 outputs)")
print(f"   Size    : {file_size:.1f} KB")

# Verification
constraint_ok = ((df['S1_mm'] > df['S2_mm']) &
                 (df['S2_mm'] > df['S3_mm']) &
                 (df['S3_mm'] > df['S4_mm'])).all()
print(f"\n   Verification:")
print(f"   Total samples       : {len(df)} {'[PASS]' if len(df) == 1000 else '[FAIL]'}")
print(f"   S1>S2>S3>S4 (radii) : {constraint_ok} {'[PASS]' if constraint_ok else '[FAIL]'}")
print(f"   No missing values   : {df.isnull().sum().sum() == 0} "
      f"{'[PASS]' if df.isnull().sum().sum() == 0 else '[FAIL]'}")

print(f"\n   Columns: {list(df.columns)}")
print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n   Statistics:")
print(df.describe().round(3).to_string())

print(f"\n{'=' * 80}")
print("   DATASET GENERATION COMPLETE!")
print(f"{'=' * 80}")
print("\nNext step: python train_models.py")
