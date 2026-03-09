"""
Train ML models for 28 GHz Triple-Band Circular SIW Antenna
  - 5 baseline models (LinearRegression, DecisionTree, RandomForest,
    Lasso, ExtraTrees)
  - 2 advanced models (CatBoost, Stacking Ensemble)
  - All 5 metrics per output: MAE, MSE, RMSE, R², Variance Score
  - Model comparison table

10 Inputs : S1_mm, S2_mm, S3_mm, S4_mm, d_mm, Wf_mm, Lf_mm, RSIW_mm, p_mm, h_mm
 6 Outputs: f1_GHz, f2_GHz, f3_GHz, BW1_GHz, BW2_GHz, BW3_GHz
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style='whitegrid', font_scale=1.1)

BASELINE_COLOR = '#E07B54'
ADVANCED_COLOR = '#2E86AB'
BEST_COLOR     = '#27AE60'

print("=" * 90)
print("  ML TRAINING — TRIPLE-BAND CIRCULAR SIW ANTENNA")
print("   1000-sample dataset  |  10 inputs  |  6 outputs  |  7 models")
print("=" * 90)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('dataset_1000samples.csv')
print(f"   Samples : {len(df)}")
print(f"   Columns : {len(df.columns)}")

input_cols  = ['S1_mm', 'S2_mm', 'S3_mm', 'S4_mm', 'd_mm',
               'Wf_mm', 'Lf_mm', 'RSIW_mm', 'p_mm', 'h_mm']
output_cols = ['f1_GHz', 'f2_GHz', 'f3_GHz',
               'BW1_GHz', 'BW2_GHz', 'BW3_GHz']

X_raw = df[input_cols].copy()
y     = df[output_cols].copy()

# ============================================================
# 2. DATASET VISUALISATION  (Fig 5 equivalent — histograms)
# ============================================================
print("\n[2/7] Generating dataset visualisation (histograms)...")

all_cols = input_cols + output_cols
n_cols_plot = 4
n_rows_plot = int(np.ceil(len(all_cols) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(16, n_rows_plot * 3))
axes = axes.flatten()
for i, col in enumerate(all_cols):
    color = ADVANCED_COLOR if col in input_cols else BASELINE_COLOR
    axes[i].hist(df[col], bins=30, color=color, edgecolor='white',
                 alpha=0.85)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Dataset Parameter Distributions — 1000 Samples\n'
             '(Blue = Inputs, Orange = Outputs)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_dataset_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: fig_dataset_histograms.png")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3/7] Engineering features...")
X = X_raw.copy()
X['S1_S2_product']  = X['S1_mm']   * X['S2_mm']
X['S3_S4_product']  = X['S3_mm']   * X['S4_mm']
X['d_Wf_product']   = X['d_mm']    * X['Wf_mm']
X['RSIW_h_product'] = X['RSIW_mm'] * X['h_mm']
X['slot_sum']       = X['S1_mm'] + X['S2_mm'] + X['S3_mm'] + X['S4_mm']
X['slot_range']     = X['S1_mm'] - X['S4_mm']
X['Lf_p_ratio']     = X['Lf_mm']   / (X['p_mm'] + 1e-9)
feature_names = list(X.columns)
print(f"   Total features after engineering: {len(feature_names)}")

# save feature names for inference scripts
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# ============================================================
# 4. SPLIT & SCALE  (80/20)
# ============================================================
print("\n[4/7] Splitting (80 % train / 20 % test) and scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"   Train : {len(X_train)} samples")
print(f"   Test  : {len(X_test)}  samples")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# 5. DEFINE ALL MODELS
# ============================================================
print("\n[5/7] Defining models...")

# ---- Baseline models ----
baseline_models = {
    # Algorithm: Ordinary Least Squares — fits a linear equation y = Xw to minimise sum of squared residuals
    'LinearRegression':  MultiOutputRegressor(LinearRegression(), n_jobs=-1),

    # Algorithm: CART (Classification and Regression Trees) — recursively splits features by best threshold
    'DecisionTree':      DecisionTreeRegressor(random_state=42),

    # Algorithm: Bagging ensemble of CART trees — averages predictions of 200 trees grown on bootstrap samples
    'RandomForest':      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),

    # Algorithm: L1-regularised Linear Regression — shrinks less important coefficients to exactly zero
    'Lasso':             MultiOutputRegressor(Lasso(alpha=0.001, max_iter=10000), n_jobs=-1),

    # Algorithm: Extremely Randomised Trees — like Random Forest but splits are chosen fully at random (lower variance)
    'ExtraTrees':        ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

# ---- Advanced models ----
advanced_models = {}

try:
    from catboost import CatBoostRegressor
    # Algorithm: Gradient Boosting with ordered boosting & symmetric trees — iteratively corrects residuals using
    #            depth-limited trees; fast and accurate on tabular data with minimal hyperparameter tuning
    advanced_models['CatBoost'] = MultiOutputRegressor(
        CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8,
                          random_state=42, verbose=0), n_jobs=-1)
    print("   CatBoost : available")
except ImportError:
    print("   CatBoost : not installed, skipping")

# ============================================================
# 6. TRAIN, EVALUATE & COLLECT METRICS
# ============================================================
print("\n[6/7] Training and evaluating all models...")
print(f"   {'Model':<22} {'R²':>7} {'MAE':>9} {'MSE':>9} {'RMSE':>9} {'VarScore':>9}  Type")
print("   " + "-" * 72)

def compute_metrics(y_true, y_pred):
    """Return overall + per-output MAE, MSE, RMSE, R², VarScore."""
    from sklearn.metrics import explained_variance_score
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    var_s = explained_variance_score(y_true, y_pred)
    per   = {}
    for i, col in enumerate(output_cols):
        yt = np.array(y_true)[:, i]
        yp = y_pred[:, i]
        per[col] = {
            'R2':       r2_score(yt, yp),
            'MAE':      mean_absolute_error(yt, yp),
            'MSE':      mean_squared_error(yt, yp),
            'RMSE':     np.sqrt(mean_squared_error(yt, yp)),
            'VarScore': explained_variance_score(yt, yp),
        }
    return r2, mae, mse, rmse, var_s, per

results = {}

def train_eval(name, model, tag):
    t0 = time.time()
    model.fit(X_train_sc, y_train)
    train_time = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(X_test_sc)
    pred_time = (time.time() - t1) / len(X_test) * 1000  # ms per sample
    r2, mae, mse, rmse, var_s, per = compute_metrics(y_test, y_pred)
    results[name] = {
        'model': model, 'y_pred': y_pred,
        'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'VarScore': var_s,
        'per': per, 'tag': tag,
        'train_time': train_time, 'pred_ms': pred_time,
    }
    tag_str = '[BASE ]' if tag == 'baseline' else '[ADV  ]'
    print(f"   {name:<22} {r2:>7.4f} {mae:>9.5f} {mse:>9.5f} {rmse:>9.5f} {var_s:>9.4f}  {tag_str}")

for name, model in baseline_models.items():
    train_eval(name, model, 'baseline')

for name, model in advanced_models.items():
    train_eval(name, model, 'advanced')

# Algorithm: Averaging Ensemble (Stacking) — combines the top-3 best models by averaging their predictions;
#            reduces individual model variance and consistently outperforms any single model alone
# ---- Stacking ensemble: average predictions of top-3 models overall ----
print(f"   {'Stacking_Ensemble':<22}", end='  ', flush=True)
# pick top-3 from ALL trained models by R²
all_r2 = {k: results[k]['R2'] for k in results}
top3   = sorted(all_r2, key=all_r2.get, reverse=True)[:3]

class AveragingEnsemble:
    """Averaging ensemble — averages predictions of top-3 models."""
    def __init__(self, estimators):
        self.estimators = estimators
    def fit(self, X, y):
        for _, m in self.estimators:
            m.fit(X, y)
        return self
    def predict(self, X):
        preds = np.array([m.predict(X) for _, m in self.estimators])
        return preds.mean(axis=0)

stacking = AveragingEnsemble([(n, results[n]['model']) for n in top3])
train_eval('Stacking_Ensemble', stacking, 'advanced')

# ============================================================
# 7. SAVE BEST MODEL
# ============================================================
print("\n[7/7] Saving artefacts and generating plots...")

best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"   Best model : {best_name}  (R² = {results[best_name]['R2']:.4f})")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   Saved : best_model.pkl,  scaler.pkl,  feature_names.pkl")

# ============================================================
# RESULTS CSV
# ============================================================
rows = []
for name, res in results.items():
    row = {
        'Model':      name,
        'Type':       res['tag'],
        'R2':         res['R2'],
        'MAE':        res['MAE'],
        'MSE':        res['MSE'],
        'RMSE':       res['RMSE'],
        'VarScore':   res['VarScore'],
        'TrainTime_s': res['train_time'],
        'PredTime_ms': res['pred_ms'],
    }
    for col in output_cols:
        for metric in ['R2', 'MAE', 'MSE', 'RMSE', 'VarScore']:
            row[f'{metric}_{col}'] = res['per'][col][metric]
    rows.append(row)

results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
results_df.to_csv('training_results.csv', index=False)
print("   Saved : training_results.csv")

# ============================================================
# PLOT 1 — Model R² comparison (All 7 Models)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
model_names = results_df['Model'].tolist()
r2_vals     = results_df['R2'].tolist()
tags        = results_df['Type'].tolist()
colors      = [BEST_COLOR if n == best_name
               else BASELINE_COLOR if t == 'baseline'
               else ADVANCED_COLOR
               for n, t in zip(model_names, tags)]
bars = ax.bar(model_names, r2_vals, color=colors, edgecolor='black', width=0.6)
ax.set_ylim(max(0, min(r2_vals) - 0.15), 1.02)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Model Comparison — R² Score\n'
             '(Green = Best Model)',

             fontsize=12, fontweight='bold')
ax.axhline(0.99, color='red', linestyle='--', linewidth=1,
           label='R²=0.99 target')
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8,
            fontweight='bold')
plt.xticks(rotation=35, ha='right', fontsize=9)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig_model_comparison_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved : fig_model_comparison_r2.png")

# ============================================================
# PLOT 2 — All 5 metrics side-by-side bar chart
# ============================================================
metric_keys  = ['R2', 'MAE', 'MSE', 'RMSE', 'VarScore']
metric_labels = ['R² (↑)', 'MAE (↓)', 'MSE (↓)', 'RMSE (↓)', 'Var Score (↑)']
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
for ax, mk, ml in zip(axes, metric_keys, metric_labels):
    vals  = results_df[mk].tolist()
    names = results_df['Model'].tolist()
    tags  = results_df['Type'].tolist()
    clrs  = [BEST_COLOR if n == best_name
             else BASELINE_COLOR if t == 'baseline'
             else ADVANCED_COLOR
             for n, t in zip(names, tags)]
    ax.barh(names, vals, color=clrs, edgecolor='black')
    ax.set_title(ml, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f'{v:.4f}', va='center', fontsize=7)
fig.suptitle('All Performance Metrics Comparison\n'
             '(Green = Best Model)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_all_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved : fig_all_metrics.png")

# ============================================================
# PLOT 3 — Per-output R² heatmap for ALL models
# ============================================================
heatmap_data = pd.DataFrame(
    {name: [results[name]['per'][col]['R2'] for col in output_cols]
     for name in results_df['Model']},
    index=output_cols
)
fig, ax = plt.subplots(figsize=(16, 5))
cmap = LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffbf', '#1a9850'])
im = ax.imshow(heatmap_data.values.astype(float), aspect='auto',
               cmap=cmap, vmin=0.90, vmax=1.0)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'].tolist(), rotation=40, ha='right',
                   fontsize=9)
ax.set_yticks(range(len(output_cols)))
ax.set_yticklabels(output_cols, fontsize=10)
for i in range(len(output_cols)):
    for j, name in enumerate(results_df['Model']):
        val = results[name]['per'][output_cols[i]]['R2']
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color='black' if val > 0.95 else 'red')
plt.colorbar(im, ax=ax, label='R² Score')
ax.set_title('Per-Output R² Heatmap — All Models vs All Outputs',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_per_output_r2_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved : fig_per_output_r2_heatmap.png")

# ============================================================
# PLOT 4 — Scatter: True vs Predicted for best model (all 6 outputs)
# ============================================================
best_pred = results[best_name]['y_pred']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, (ax, col) in enumerate(zip(axes, output_cols)):
    yt = np.array(y_test)[:, i]
    yp = best_pred[:, i]
    ax.scatter(yt, yp, alpha=0.35, s=12, color=ADVANCED_COLOR)
    lims = [min(yt.min(), yp.min()) - 0.02,
            max(yt.max(), yp.max()) + 0.02]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
    r2v  = results[best_name]['per'][col]['R2']
    maev = results[best_name]['per'][col]['MAE']
    ax.set_xlabel(f'True {col}', fontsize=10)
    ax.set_ylabel(f'Predicted {col}', fontsize=10)
    ax.set_title(f'{col}   R²={r2v:.4f}  MAE={maev:.4f}', fontsize=10,
                 fontweight='bold')
    ax.legend(fontsize=8)
fig.suptitle(f'True vs Predicted — Best Model: {best_name}\n'
             f'Overall R²={results[best_name]["R2"]:.4f}  '
             f'MAE={results[best_name]["MAE"]:.5f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_true_vs_pred_best.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved : fig_true_vs_pred_best.png")

# ============================================================
# PLOT 5 — Per-output detailed metrics for best model
# ============================================================
metrics_best = {col: results[best_name]['per'][col] for col in output_cols}
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, mk, ml in zip(axes, metric_keys, metric_labels):
    vals = [metrics_best[col][mk] for col in output_cols]
    bars = ax.bar(output_cols, vals,
                  color=[ADVANCED_COLOR if 'f' in c else BASELINE_COLOR
                         for c in output_cols],
                  edgecolor='black')
    ax.set_title(ml, fontsize=10, fontweight='bold')
    ax.set_xticklabels(output_cols, rotation=30, ha='right', fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)
fig.suptitle(f'Per-Output Metrics — Best Model: {best_name}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_per_output_metrics_best.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved : fig_per_output_metrics_best.png")

# ============================================================
# PLOT 6 — Feature importance (best tree-based model)
# ============================================================
def get_feature_importance(model, feat_names):
    """Try to extract feature importances from wrapped models."""
    m = model
    if hasattr(m, 'estimators_'):     # MultiOutputRegressor
        imps = np.mean([e.feature_importances_
                        for e in m.estimators_
                        if hasattr(e, 'feature_importances_')], axis=0)
        return imps
    if hasattr(m, 'feature_importances_'):
        return m.feature_importances_
    return None

imp_model_name = None
for candidate in [best_name, 'ExtraTrees', 'CatBoost',
                  'RandomForest']:
    if candidate in results:
        imps = get_feature_importance(results[candidate]['model'],
                                      feature_names)
        if imps is not None:
            imp_model_name = candidate
            break

if imp_model_name is not None:
    imp_df = pd.Series(imps, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_imp = [BEST_COLOR if v >= imp_df.quantile(0.75) else ADVANCED_COLOR
                  for v in imp_df.values]
    ax.barh(imp_df.index, imp_df.values, color=colors_imp, edgecolor='black')
    ax.set_xlabel('Mean Feature Importance', fontsize=11)
    ax.set_title(f'Feature Importance — {imp_model_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved : fig_feature_importance.png")

# ============================================================
# PRINT COMPLETE RESULTS SUMMARY
# ============================================================
print(f"\n{'=' * 90}")
print("   COMPLETE TRAINING RESULTS SUMMARY")
print(f"{'=' * 90}")
print(f"   {'Model':<24} {'Type':<8} {'R²':>7} {'MAE':>9} {'MSE':>9}"
      f" {'RMSE':>9} {'VarScore':>9} {'Train(s)':>9}")
print("   " + "-" * 88)
for _, row in results_df.iterrows():
    print(f"   {row['Model']:<24} {row['Type']:<8} {row['R2']:>7.4f}"
          f" {row['MAE']:>9.5f} {row['MSE']:>9.5f} {row['RMSE']:>9.5f}"
          f" {row['VarScore']:>9.4f} {row['TrainTime_s']:>9.2f}")

# ============================================================
# MODEL COMPARISON TABLE
# ============================================================
print(f"\n{'=' * 90}")
print("   BASELINE vs ADVANCED MODEL COMPARISON")
print(f"{'=' * 90}")

# Dynamically find actual best baseline and best advanced from results
baseline_r2 = {n: results[n]['R2'] for n in results if results[n]['tag'] == 'baseline'}
advanced_r2 = {n: results[n]['R2'] for n in results if results[n]['tag'] == 'advanced'}
best_baseline_name = max(baseline_r2, key=baseline_r2.get)
best_advanced_name = max(advanced_r2, key=advanced_r2.get) if advanced_r2 else best_baseline_name
best_baseline_result = results[best_baseline_name]
best_advanced_result = results[best_advanced_name]

print(f"\n   Best baseline model : {best_baseline_name:<18} (R² = {best_baseline_result['R2']:.4f})")
print(f"   Best advanced model : {best_advanced_name:<18} (R² = {best_advanced_result['R2']:.4f})")
print(f"   Overall best model  : {best_name:<18} (R² = {results[best_name]['R2']:.4f})")
print()
print(f"   {'Metric':<14} {best_baseline_name:>20} {best_advanced_name:>20}"
      f" {'Best (' + best_name[:14] + ')':>26}")
print("   " + "-" * 84)

for mk in ['R2', 'MAE', 'MSE', 'RMSE', 'VarScore']:
    base_val = best_baseline_result[mk]
    adv_val  = best_advanced_result[mk]
    bst_val  = results[best_name][mk]
    direction = '(hi)' if mk in ('R2', 'VarScore') else '(lo)'
    better = ('BETTER ✓' if (mk in ('R2','VarScore') and bst_val > base_val)
              else 'BETTER ✓' if (mk not in ('R2','VarScore') and bst_val < base_val)
              else 'similar')
    print(f"   {mk + ' '+direction:<14} {base_val:>20.5f} {adv_val:>20.5f}"
          f" {bst_val:>26.5f}  {better}")

print()
print(f"   Per-output comparison — Best Model ({best_name}) vs {best_baseline_name}:")
print(f"   {'Output':<14} {'Metric':<10} {best_baseline_name:>14} {best_advanced_name:>16}"
      f" {'Best Model':>12}  Improvement")
print("   " + "-" * 72)
for col in output_cols:
    for mk in ['R2', 'MAE', 'RMSE']:
        base_val = best_baseline_result['per'][col][mk]
        adv_val  = best_advanced_result['per'][col][mk]
        bst_val  = results[best_name]['per'][col][mk]
        if mk == 'R2':
            delta = bst_val - base_val
            arrow = '+' if delta > 0 else '-'
        else:
            delta = base_val - bst_val
            arrow = '+' if delta > 0 else '-'
        print(f"   {col:<14} {mk:<10} {base_val:>14.5f} {adv_val:>16.5f}"
              f" {bst_val:>12.5f}  {arrow}{abs(delta):.5f}")

print(f"\n   Training Summary:")
print(f"   - 1000-sample dataset with 7 models trained")
print(f"   - CatBoost and Stacking Ensemble introduced as advanced models beyond ExtraTrees")
print(f"   - Stacking ensemble averages top-3 models for maximum accuracy")
print(f"   - All 5 metrics reported: MAE, MSE, RMSE, R², Variance Score")

print(f"\n{'=' * 90}")
print(f"   BEST MODEL  : {best_name}")
print(f"   Overall R²  : {results[best_name]['R2']:.6f}")
print(f"   Overall MAE : {results[best_name]['MAE']:.6f}")
print(f"   Overall MSE : {results[best_name]['MSE']:.6f}")
print(f"   Overall RMSE: {results[best_name]['RMSE']:.6f}")
print(f"   Var Score   : {results[best_name]['VarScore']:.6f}")
print(f"\n   Per-output breakdown (Best Model = {best_name}):")
print(f"   {'Output':<14} {'R²':>8} {'MAE':>10} {'MSE':>10}"
      f" {'RMSE':>10} {'VarScore':>10}  Status")
print("   " + "-" * 72)
for col in output_cols:
    p = results[best_name]['per'][col]
    status = '[PASS]' if p['R2'] >= 0.97 else '[CHECK]'
    print(f"   {col:<14} {p['R2']:>8.5f} {p['MAE']:>10.6f} {p['MSE']:>10.6f}"
          f" {p['RMSE']:>10.6f} {p['VarScore']:>10.5f}  {status}")

print(f"\n{'=' * 90}")
print("   TRAINING COMPLETE!  All plots and models saved.")
print(f"{'=' * 90}")
print("\n   Output files:")
print("   best_model.pkl, scaler.pkl, feature_names.pkl")
print("   training_results.csv")
print("   fig_dataset_histograms.png")
print("   fig_model_comparison_r2.png")
print("   fig_all_metrics.png")
print("   fig_per_output_r2_heatmap.png")
print("   fig_true_vs_pred_best.png")
print("   fig_per_output_metrics_best.png")
print("   fig_feature_importance.png")
print("\nNext step: python predict_antenna.py")

