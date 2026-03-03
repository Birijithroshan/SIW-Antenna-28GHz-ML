"""
Generate 1000-sample dataset for 28 GHz Triple-Band Circular SIW Antenna
6 Inputs: S1, S2, S3, S4, d_via, Wf
6 Outputs: Freq1_GHz, Freq2_GHz, Freq3_GHz, Bandwidth1_MHz, Bandwidth2_MHz, Bandwidth3_MHz
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

# Via diameter and feed width
d_via = np.random.uniform(0.45, 0.65, n_samples)
Wf = np.random.uniform(1.00, 1.20, n_samples)

print(f"   S1:    {S1.min():.2f} - {S1.max():.2f} mm")
print(f"   S2:    {S2.min():.2f} - {S2.max():.2f} mm")
print(f"   S3:    {S3.min():.2f} - {S3.max():.2f} mm")
print(f"   S4:    {S4.min():.2f} - {S4.max():.2f} mm")
print(f"   d_via: {d_via.min():.2f} - {d_via.max():.2f} mm")
print(f"   Wf:    {Wf.min():.2f} - {Wf.max():.2f} mm")

# ========================================
# FREQUENCY CALCULATIONS (Physics-based)
# ========================================
print(f"\n[2/5] Calculating triple-band resonant frequencies...")

# Freq1 (K-band: ~26.5-28.5 GHz)
f1_base = 27.0
f1 = (f1_base
      + 0.35 * (S1 - 6.0)
      + 0.28 * (S2 - 5.0)
      + 0.20 * (S3 - 4.0)
      + 0.15 * (S4 - 3.0)
      + 0.40 * (d_via - 0.55)
      + 0.25 * (Wf - 1.10)
      + 0.08 * (S1 * S2 / 30.0)
      + 0.05 * (d_via * Wf)
      + 0.02 * ((S1 - 6.0) ** 2)
      + np.random.normal(0, 0.015, n_samples))
Freq1_GHz = np.clip(np.round(f1, 2), 26.50, 28.50)

# Freq2 (Ka-band lower: ~31-33 GHz)
f2_base = 31.80
f2 = (f2_base
      + 0.40 * (S1 - 6.0)
      + 0.32 * (S2 - 5.0)
      + 0.25 * (S3 - 4.0)
      + 0.20 * (S4 - 3.0)
      + 0.35 * (d_via - 0.55)
      + 0.22 * (Wf - 1.10)
      + 0.06 * (S1 * S3 / 25.0)
      + 0.04 * (d_via * Wf)
      + np.random.normal(0, 0.018, n_samples))
Freq2_GHz = np.clip(np.round(f2, 2), 30.50, 33.00)

# Freq3 (Ka-band upper: ~33-35.5 GHz)
f3_base = 34.00
f3 = (f3_base
      + 0.45 * (S1 - 6.0)
      + 0.35 * (S2 - 5.0)
      + 0.28 * (S3 - 4.0)
      + 0.22 * (S4 - 3.0)
      + 0.30 * (d_via - 0.55)
      + 0.18 * (Wf - 1.10)
      + 0.07 * (S2 * S4 / 16.0)
      + 0.03 * (d_via * Wf)
      + np.random.normal(0, 0.020, n_samples))
Freq3_GHz = np.clip(np.round(f3, 2), 32.80, 35.50)

print(f"   Freq1: {Freq1_GHz.min():.2f} - {Freq1_GHz.max():.2f} GHz (K-band)")
print(f"   Freq2: {Freq2_GHz.min():.2f} - {Freq2_GHz.max():.2f} GHz (Ka-band lower)")
print(f"   Freq3: {Freq3_GHz.min():.2f} - {Freq3_GHz.max():.2f} GHz (Ka-band upper)")

# ========================================
# BANDWIDTH CALCULATIONS (Physics-based)
# ========================================
print(f"\n[3/5] Calculating triple-band bandwidths...")

# BW1 (for Freq1)
bw1 = (1950
       + 120 * (Freq1_GHz - 27.0)
       + 220 * (Wf - 1.10)
       + 60 * (S1 - 6.0)
       + 40 * (S2 - 5.0)
       + 50 * (d_via - 0.55)
       + 35 * (Wf * (Freq1_GHz - 27.0))
       + np.random.normal(0, 10, n_samples))
Bandwidth1_MHz = np.clip(np.round(bw1, 0), 1800, 2150)

# BW2 (for Freq2 - typically wider)
bw2 = (2550
       + 100 * (Freq2_GHz - 31.80)
       + 200 * (Wf - 1.10)
       + 55 * (S1 - 6.0)
       + 45 * (S3 - 4.0)
       + 40 * (d_via - 0.55)
       + 30 * (Wf * (Freq2_GHz - 31.80))
       + np.random.normal(0, 12, n_samples))
Bandwidth2_MHz = np.clip(np.round(bw2, 0), 2400, 2750)

# BW3 (for Freq3)
bw3 = (2100
       + 110 * (Freq3_GHz - 34.0)
       + 210 * (Wf - 1.10)
       + 50 * (S2 - 5.0)
       + 45 * (S4 - 3.0)
       + 45 * (d_via - 0.55)
       + 32 * (Wf * (Freq3_GHz - 34.0))
       + np.random.normal(0, 11, n_samples))
Bandwidth3_MHz = np.clip(np.round(bw3, 0), 1950, 2300)

print(f"   BW1: {Bandwidth1_MHz.min():.0f} - {Bandwidth1_MHz.max():.0f} MHz")
print(f"   BW2: {Bandwidth2_MHz.min():.0f} - {Bandwidth2_MHz.max():.0f} MHz")
print(f"   BW3: {Bandwidth3_MHz.min():.0f} - {Bandwidth3_MHz.max():.0f} MHz")

# ========================================
# CREATE DATAFRAME
# ========================================
print(f"\n[4/5] Building DataFrame...")


df = pd.DataFrame({
    'S1': np.round(S1, 2),
    'S2': np.round(S2, 2),
    'S3': np.round(S3, 2),
    'S4': np.round(S4, 2),
    'd_via': np.round(d_via, 2),
    'Wf': np.round(Wf, 2),
    'Freq1_GHz': Freq1_GHz,
    'Freq2_GHz': Freq2_GHz,
    'Freq3_GHz': Freq3_GHz,
    'Bandwidth1_MHz': Bandwidth1_MHz.astype(int),
    'Bandwidth2_MHz': Bandwidth2_MHz.astype(int),
    'Bandwidth3_MHz': Bandwidth3_MHz.astype(int)
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
print(f"   File: {output_file}")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Size: {file_size:.1f} KB")

# Verification
constraint_ok = ((df['S1'] > df['S2']) & (df['S2'] > df['S3']) & (df['S3'] > df['S4'])).all()
print(f"\n   Verification:")
print(f"   Total samples: {len(df)} {'[PASS]' if len(df) == 1000 else '[FAIL]'}")
print(f"   S1 > S2 > S3 > S4: {constraint_ok} {'[PASS]' if constraint_ok else '[FAIL]'}")
print(f"   No missing values: {df.isnull().sum().sum() == 0} {'[PASS]' if df.isnull().sum().sum() == 0 else '[FAIL]'}")

print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n   Statistics:")
print(df.describe().round(2).to_string())

print(f"\n{'=' * 80}")
print("   DATASET GENERATION COMPLETE!")
print(f"{'=' * 80}")
print("\nNext step: python train_models.py")
