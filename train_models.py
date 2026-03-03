"""
Train ML models for 28 GHz Triple-Band Circular SIW Antenna
6 Inputs: S1, S2, S3, S4, d_via, Wf
6 Outputs: Freq1_GHz, Freq2_GHz, Freq3_GHz, Bandwidth1_MHz, Bandwidth2_MHz, Bandwidth3_MHz
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               GradientBoostingRegressor, StackingRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

print("=" * 80)
print("      TRAINING ML MODELS - TRIPLE-BAND SIW ANTENNA")
print("=" * 80)

# ========================================
# LOAD DATA
# ========================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('dataset_1000samples.csv')
print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")

input_cols = ['S1', 'S2', 'S3', 'S4', 'd_via', 'Wf']
output_cols = ['Freq1_GHz', 'Freq2_GHz', 'Freq3_GHz',
               'Bandwidth1_MHz', 'Bandwidth2_MHz', 'Bandwidth3_MHz']

X = df[input_cols].copy()
y = df[output_cols].copy()

# ========================================
# FEATURE ENGINEERING
# ========================================
print("\n[2/6] Engineering features...")
X['S1_S2_product'] = X['S1'] * X['S2']
X['S3_S4_product'] = X['S3'] * X['S4']
X['dvia_Wf_product'] = X['d_via'] * X['Wf']
X['slot_sum'] = X['S1'] + X['S2'] + X['S3'] + X['S4']
X['slot_range'] = X['S1'] - X['S4']
print(f"   Features: {list(X.columns)} ({len(X.columns)} total)")

# ========================================
# SPLIT & SCALE
# ========================================
print("\n[3/6] Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ========================================
# DEFINE MODELS
# ========================================
print("\n[4/6] Training models...")

models = {
    'ExtraTrees': ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'GradientBoosting': MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, random_state=42), n_jobs=-1),
}

try:
    from xgboost import XGBRegressor
    models['XGBoost'] = MultiOutputRegressor(
        XGBRegressor(n_estimators=200, random_state=42, verbosity=0), n_jobs=-1)
    print("   XGBoost available")
except ImportError:
    print("   XGBoost not installed, skipping")

try:
    from lightgbm import LGBMRegressor
    models['LightGBM'] = MultiOutputRegressor(
        LGBMRegressor(n_estimators=200, random_state=42, verbose=-1), n_jobs=-1)
    print("   LightGBM available")
except ImportError:
    print("   LightGBM not installed, skipping")

try:
    from catboost import CatBoostRegressor
    models['CatBoost'] = MultiOutputRegressor(
        CatBoostRegressor(iterations=200, random_state=42, verbose=0), n_jobs=-1)
    print("   CatBoost available")
except ImportError:
    print("   CatBoost not installed, skipping")

# ========================================
# TRAIN & EVALUATE
# ========================================
results = {}
for name, model in models.items():
    print(f"   Training {name}...", end=' ', flush=True)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    per_output_r2 = {}
    for i, col in enumerate(output_cols):
        per_output_r2[col] = r2_score(y_test.iloc[:, i], y_pred[:, i])

    results[name] = {
        'model': model, 'y_pred': y_pred,
        'R2': r2, 'MAE': mae, 'RMSE': rmse,
        'per_output_r2': per_output_r2
    }
    print(f"R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

# ========================================
# STACKING ENSEMBLE (wrapped for multi-output)
# ========================================
print("   Building stacking ensemble...")
stacking = MultiOutputRegressor(
    StackingRegressor(
        estimators=[
            ('et', ExtraTreesRegressor(n_estimators=200, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ],
        final_estimator=Ridge(),
        cv=5,
    ),
    n_jobs=-1,
)
stacking.fit(X_train_sc, y_train)
y_pred_stack = stacking.predict(X_test_sc)
r2_s = r2_score(y_test, y_pred_stack)
mae_s = mean_absolute_error(y_test, y_pred_stack)
rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_stack))
per_output_r2_s = {col: r2_score(y_test.iloc[:, i], y_pred_stack[:, i])
                   for i, col in enumerate(output_cols)}
results['Stacking'] = {
    'model': stacking, 'y_pred': y_pred_stack,
    'R2': r2_s, 'MAE': mae_s, 'RMSE': rmse_s,
    'per_output_r2': per_output_r2_s
}
print(f"   Stacking        R²={r2_s:.4f}  MAE={mae_s:.4f}  RMSE={rmse_s:.4f}")

# ========================================
# SAVE BEST MODEL
# ========================================
print("\n[5/6] Saving best model...")
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"   Best model: {best_name} (R²={results[best_name]['R2']:.4f})")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   Saved: best_model.pkl, scaler.pkl")

# ========================================
# SAVE RESULTS CSV
# ========================================
rows = []
for name, res in results.items():
    row = {'Model': name, 'R2': res['R2'], 'MAE': res['MAE'], 'RMSE': res['RMSE']}
    row.update({f'R2_{k}': v for k, v in res['per_output_r2'].items()})
    rows.append(row)
results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
results_df.to_csv('training_results.csv', index=False)
print("   Saved: training_results.csv")

# ========================================
# PLOTS
# ========================================
print("\n[6/6] Generating plots...")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = results_df['Model'].tolist()
r2_vals = results_df['R2'].tolist()
colors = ['green' if n == best_name else 'steelblue' for n in names]
bars = ax.bar(names, r2_vals, color=colors, edgecolor='black')
ax.set_ylim(max(0, min(r2_vals) - 0.05), 1.0)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison - Overall R² Score')
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()
print("   Saved: model_comparison.png")

# Bandwidth predictions scatter
best_pred = results[best_name]['y_pred']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
bw_cols = ['Bandwidth1_MHz', 'Bandwidth2_MHz', 'Bandwidth3_MHz']
for i, (ax, col) in enumerate(zip(axes, bw_cols)):
    idx = output_cols.index(col)
    true_vals = y_test.iloc[:, idx]
    pred_vals = best_pred[:, idx]
    ax.scatter(true_vals, pred_vals, alpha=0.4, s=15, color='steelblue')
    lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel(f'True {col}')
    ax.set_ylabel(f'Predicted {col}')
    r2_bw = r2_score(true_vals, pred_vals)
    ax.set_title(f'{col}\nR²={r2_bw:.4f}')
fig.suptitle(f'Bandwidth Predictions - {best_name}', fontsize=13)
plt.tight_layout()
plt.savefig('bandwidth_predictions.png', dpi=150)
plt.close()
print("   Saved: bandwidth_predictions.png")

# ========================================
# DETAILED RESULTS
# ========================================
print(f"\n{'=' * 80}")
print("   TRAINING RESULTS SUMMARY")
print(f"{'=' * 80}")
print(results_df[['Model', 'R2', 'MAE', 'RMSE']].to_string(index=False))

print(f"\n   Per-output R² for best model ({best_name}):")
for col, val in results[best_name]['per_output_r2'].items():
    status = '[PASS]' if val >= 0.95 else '[CHECK]'
    print(f"   {col:25s}: {val:.4f}  {status}")

print(f"\n   Verification:")
print(f"   Best model: {best_name}")
print(f"   Overall R²: {results[best_name]['R2']:.4f}  "
      f"{'[PASS]' if results[best_name]['R2'] >= 0.95 else '[CHECK]'}")
print(f"\n{'=' * 80}")
print("   TRAINING COMPLETE!")
print(f"{'=' * 80}")
print("\nNext step: python predict_antenna.py")
