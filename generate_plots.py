"""
Publication Plot Generator -- Triple-Band Circular SIW Antenna ML Project
==========================================================================
Models: LinearRegression, DecisionTree, ExtraTrees, Lasso, RandomForest,
        CatBoost, Stacking_Ensemble  (7 total)

Figures generated:
  Fig 1 -- Histograms of all 16 dataset parameters
  Fig 2 -- Error & Accuracy for all 7 models (2x2)
  Fig 3 -- Actual vs Predicted resonance frequencies, all 7 models (2x4)
  Fig 4 -- Actual vs Predicted bandwidths, all 7 models (2x4)
  Fig 5 -- Full comparison: all 7 models side-by-side (R2 and MAE)
  Fig 6 -- Key comparison: ExtraTrees vs CatBoost vs Stacking_Ensemble
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, explained_variance_score)

try:
    from catboost import CatBoostRegressor
    _CAT = True
except ImportError:
    _CAT = False
    print("WARNING: catboost not installed -- CatBoost will be skipped")

# ---- Style -------------------------------------------------------------------
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       130,
})

BASELINE_COLOR   = '#2E86AB'
ADVANCED_COLOR   = '#E07B54'
BASELINE_PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
ADVANCED_PALETTE = ['#ffbb78','#98df8a']
FREQ_PALETTE  = ['#e41a1c','#377eb8','#4daf4a']
BW_PALETTE    = ['#ff7f00','#a65628','#f781bf']
INPUT_COLOR   = '#2E86AB'
OUTPUT_COLOR  = '#E07B54'

print("=" * 65)
print("  GENERATING PUBLICATION PLOTS -- TRIPLE-BAND SIW ANTENNA")
print("=" * 65)

# ==============================================================================
# 1. LOAD DATA & FEATURE ENGINEERING
# ==============================================================================
print("\n[1/4] Loading dataset and engineering features...")

df = pd.read_csv('dataset_1000samples.csv')

input_cols  = ['S1_mm','S2_mm','S3_mm','S4_mm','d_mm',
               'Wf_mm','Lf_mm','RSIW_mm','p_mm','h_mm']
output_cols = ['f1_GHz','f2_GHz','f3_GHz','BW1_GHz','BW2_GHz','BW3_GHz']
freq_cols   = ['f1_GHz','f2_GHz','f3_GHz']
bw_cols     = ['BW1_GHz','BW2_GHz','BW3_GHz']

X = df[input_cols].copy()
y = df[output_cols].copy()

X['S1_S2_product']  = X['S1_mm']   * X['S2_mm']
X['S3_S4_product']  = X['S3_mm']   * X['S4_mm']
X['d_Wf_product']   = X['d_mm']    * X['Wf_mm']
X['RSIW_h_product'] = X['RSIW_mm'] * X['h_mm']
X['slot_sum']       = X['S1_mm'] + X['S2_mm'] + X['S3_mm'] + X['S4_mm']
X['slot_range']     = X['S1_mm'] - X['S4_mm']
X['Lf_p_ratio']     = X['Lf_mm']   / (X['p_mm'] + 1e-9)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
y_np       = y_test.values   # shape (200, 6)

# ==============================================================================
# 2. DEFINE MODELS
# ==============================================================================

class AveragingEnsemble:
    """Averages predictions from base estimators."""
    def __init__(self, estimators):
        self.estimators = estimators
    def fit(self, X, y):
        for _, m in self.estimators:
            m.fit(X, y)
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for _, m in self.estimators], axis=0)

baseline_models = {
    'Linear Regression': MultiOutputRegressor(LinearRegression(), n_jobs=-1),
    'Decision Tree':     DecisionTreeRegressor(random_state=42),
    'Extra Trees':       ExtraTreesRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1),
    'Lasso':             MultiOutputRegressor(Lasso(alpha=0.001, max_iter=10000),
                                              n_jobs=-1),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42,
                                               n_jobs=-1),
}
baseline_names   = list(baseline_models.keys())
baseline_abbrevs = ['LR','DT','ET','LA','RF']

advanced_models = {}
if _CAT:
    advanced_models['CatBoost'] = MultiOutputRegressor(
        CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8,
                          random_state=42, verbose=0), n_jobs=-1)
advanced_models['Stacking_Ensemble'] = AveragingEnsemble([
    ('lr', MultiOutputRegressor(LinearRegression(), n_jobs=-1)),
    ('la', MultiOutputRegressor(Lasso(alpha=0.001, max_iter=10000), n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])
advanced_names   = list(advanced_models.keys())
advanced_abbrevs = ['CAT','Stack'][:len(advanced_names)]
pal_advanced     = ADVANCED_PALETTE[:len(advanced_names)]

# ==============================================================================
# 3. TRAIN ALL MODELS
# ==============================================================================
print("[2/4] Training all 7 models...")

preds   = {}
metrics = {}

def _train_eval(name, mdl, tag):
    print(f"   [{tag}] {name}...", end=' ', flush=True)
    mdl.fit(X_train_sc, y_train)
    yp = mdl.predict(X_test_sc)
    preds[name] = yp
    per = {}
    for i, col in enumerate(output_cols):
        yt = y_np[:, i]; ypi = yp[:, i]
        per[col] = {
            'R2':       r2_score(yt, ypi),
            'MAE':      mean_absolute_error(yt, ypi),
            'MSE':      mean_squared_error(yt, ypi),
            'RMSE':     np.sqrt(mean_squared_error(yt, ypi)),
            'VarScore': explained_variance_score(yt, ypi),
        }
    metrics[name] = per
    print(f"R2={r2_score(y_np, yp):.4f}")

for n, m in baseline_models.items():
    _train_eval(n, m, 'base ')
for n, m in advanced_models.items():
    _train_eval(n, m, 'adv  ')

def avg(name, cols, key):
    return float(np.mean([metrics[name][c][key] for c in cols]))

print("[3/4] Training complete.\n")

# ==============================================================================
# 4. PLOTTING HELPERS
# ==============================================================================

def save_show(fname):
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"   Saved: {fname}")
    plt.show()

def scatter_avp(ax, name, target_cols, palette, labels,
                xlabel, ylabel, panel_lbl):
    yp  = preds[name]
    col_idxs = [output_cols.index(c) for c in target_cols]
    all_t, all_p = [], []
    for ci, col in zip(col_idxs, target_cols):
        yt = y_np[:, ci]; ypi = yp[:, ci]
        ax.scatter(yt, ypi,
                   color=palette[target_cols.index(col)],
                   alpha=0.55, s=20,
                   label=labels[target_cols.index(col)], zorder=3)
        all_t += yt.tolist(); all_p += ypi.tolist()
    at, ap = np.array(all_t), np.array(all_p)
    lo = min(at.min(), ap.min()) * 0.998
    hi = max(at.max(), ap.max()) * 1.002
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4, label='Ideal (y=x)', zorder=4)
    r2v = r2_score(at, ap)
    mae = mean_absolute_error(at, ap)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{panel_lbl} {name}\nR2 = {r2v:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax.text(0.98, 0.04, f'MAE = {mae:.4f} GHz',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

def err_bars(ax, x, v1, v2, v3, xlabs, title):
    w = 0.25
    b1 = ax.bar(x-w, v1, w, label='MAE',  color='#e41a1c', edgecolor='k')
    b2 = ax.bar(x,   v2, w, label='MSE',  color='#377eb8', edgecolor='k')
    b3 = ax.bar(x+w, v3, w, label='RMSE', color='#4daf4a', edgecolor='k')
    ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=10)
    ax.set_ylabel('Error Value (GHz)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.0002,
                    f'{h:.4f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=90)

def acc_bars(ax, x, r2v, varv, xlabs, title):
    w2 = 0.32
    b4 = ax.bar(x-w2/2, r2v,  w2, label='R2 (%)',        color='#984ea3', edgecolor='k')
    b5 = ax.bar(x+w2/2, varv, w2, label='Var Score (%)', color='#ff7f00', edgecolor='k')
    ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='red', ls='--', lw=1, alpha=0.6, label='100% ref')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for bars in [b4, b5]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

def cmp_bar(ax, xall, vals, colors, xlabs, title, ylabel, ref=None, fmt='.2f'):
    bars = ax.bar(xall, vals, color=colors, edgecolor='k', width=0.65)
    ax.set_xticks(xall)
    ax.set_xticklabels(xlabs, fontsize=10, rotation=10, ha='right')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(len(baseline_names)-0.5, color='k', lw=2, ls=':', alpha=0.6)
    if ref is not None:
        ax.axhline(ref, color='red', ls='--', lw=1.2, alpha=0.6,
                   label=f'{ref}% ideal')
        ax.legend(fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(0.3 if ref else 0.00005),
                f'{v:{fmt}}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', rotation=90)

# ==============================================================================
# FIG 1  -- Dataset parameter histograms  (4 x 4)
# ==============================================================================
print("[4/4] Drawing figures...")
print("  -> Fig 1: Dataset histograms")

all16  = input_cols + output_cols
lbl16  = ['S1 (mm)','S2 (mm)','S3 (mm)','S4 (mm)',
           'd (mm)','Wf (mm)','Lf (mm)','RSIW (mm)','p (mm)','h (mm)',
           'f1 (GHz)','f2 (GHz)','f3 (GHz)',
           'BW1 (GHz)','BW2 (GHz)','BW3 (GHz)']

fig, axes = plt.subplots(4, 4, figsize=(16, 13))
for ax, col, lbl in zip(axes.flatten(), all16, lbl16):
    color = INPUT_COLOR if col in input_cols else OUTPUT_COLOR
    data  = df[col].values
    cnt, edg = np.histogram(data, bins=25)
    ax.bar(edg[:-1], cnt, width=np.diff(edg),
           color=color, edgecolor='white', lw=0.5, alpha=0.88)
    kx = np.linspace(data.min(), data.max(), 300)
    ky = gaussian_kde(data)(kx) * len(data) * np.diff(edg).mean()
    ax.plot(kx, ky, 'k--', lw=1.5)
    ax.set_title(lbl, fontsize=10, fontweight='bold', pad=3)
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Count',  fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.text(0.97, 0.95,
            f'u={data.mean():.3f}\ns={data.std():.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

fig.legend(handles=[Patch(facecolor=INPUT_COLOR,  label='Input parameters'),
                    Patch(facecolor=OUTPUT_COLOR, label='Output parameters')],
           loc='lower center', ncol=2, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Fig. 1 -- Graphical Representation of Dataset Parameters\n'
             '(1000 Samples -- Triple-Band Circular SIW Antenna)',
             fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout(rect=[0, 0.02, 1, 1])
save_show('plot_fig1_histograms.png')

# ==============================================================================
# FIG 2  -- Error & Accuracy -- All 7 Models  (2 x 2)
# ==============================================================================
print("  -> Fig 2: All 7 models -- Error & Accuracy")

all_names   = baseline_names + advanced_names
all_abbrevs = baseline_abbrevs + advanced_abbrevs
ALL_PALETTE = BASELINE_PALETTE[:len(baseline_names)] + ADVANCED_PALETTE[:len(advanced_names)]
xall        = np.arange(len(all_names))

a_f_mae  = [avg(n, freq_cols, 'MAE')       for n in all_names]
a_f_mse  = [avg(n, freq_cols, 'MSE')       for n in all_names]
a_f_rmse = [avg(n, freq_cols, 'RMSE')      for n in all_names]
a_f_r2   = [avg(n, freq_cols, 'R2') * 100  for n in all_names]
a_f_var  = [avg(n, freq_cols, 'VarScore') * 100 for n in all_names]
a_b_mae  = [avg(n, bw_cols,   'MAE')       for n in all_names]
a_b_mse  = [avg(n, bw_cols,   'MSE')       for n in all_names]
a_b_rmse = [avg(n, bw_cols,   'RMSE')      for n in all_names]
a_b_r2   = [avg(n, bw_cols,   'R2') * 100  for n in all_names]
a_b_var  = [avg(n, bw_cols,   'VarScore') * 100 for n in all_names]

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
err_bars(axes[0,0], xall, a_f_mae, a_f_mse, a_f_rmse, all_abbrevs,
         '(a) Frequency Error -- MAE / MSE / RMSE  (lower is better)')
acc_bars(axes[0,1], xall, a_f_r2, a_f_var, all_abbrevs,
         '(b) Frequency Accuracy -- R2 and Variance %  (higher is better)')
err_bars(axes[1,0], xall, a_b_mae, a_b_mse, a_b_rmse, all_abbrevs,
         '(c) Bandwidth Error -- MAE / MSE / RMSE  (lower is better)')
acc_bars(axes[1,1], xall, a_b_r2, a_b_var, all_abbrevs,
         '(d) Bandwidth Accuracy -- R2 and Variance %  (higher is better)')

fig.legend(handles=[Line2D([0],[0], color=c, marker='s', ls='', ms=10,
                    label=f'{ab} = {nm}')
                    for c, ab, nm in zip(ALL_PALETTE, all_abbrevs, all_names)],
           loc='lower center', ncol=4, fontsize=9.5,
           frameon=True, bbox_to_anchor=(0.5, -0.03))
fig.suptitle('Fig. 2 -- ML Error and Accuracy Analysis -- All 7 Models\n'
             'Frequency (a,b) and Bandwidth (c,d)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
save_show('plot_fig2_error_accuracy.png')

# ==============================================================================
# FIG 3  -- Actual vs Predicted Resonance Frequencies -- All 7 Models  (2 x 4)
# ==============================================================================
print("  -> Fig 3: All 7 models -- Actual vs Predicted frequencies")

panels8 = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for ax, name, panel in zip(axes.flatten(), all_names, panels8):
    scatter_avp(ax, name, freq_cols, FREQ_PALETTE,
                ['f1 (~27 GHz)','f2 (~32 GHz)','f3 (~34 GHz)'],
                'Actual Resonant Frequency (GHz)',
                'Predicted Resonant Frequency (GHz)', panel)
axes.flatten()[-1].set_visible(False)
fig.suptitle('Fig. 3 -- Actual vs. Predicted Resonance Frequencies -- All 7 Models',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_show('plot_fig3_freq_scatter.png')

# ==============================================================================
# FIG 4  -- Actual vs Predicted Bandwidths -- All 7 Models  (2 x 4)
# ==============================================================================
print("  -> Fig 4: All 7 models -- Actual vs Predicted bandwidths")

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for ax, name, panel in zip(axes.flatten(), all_names, panels8):
    scatter_avp(ax, name, bw_cols, BW_PALETTE,
                ['BW1 (at f1)','BW2 (at f2)','BW3 (at f3)'],
                'Actual Bandwidth (GHz)',
                'Predicted Bandwidth (GHz)', panel)
axes.flatten()[-1].set_visible(False)
fig.suptitle('Fig. 4 -- Actual vs. Predicted Bandwidths -- All 7 Models',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_show('plot_fig4_bw_scatter.png')

# ==============================================================================
# FIG 5  -- Full model comparison: all 7 models  (2 x 2)
# ==============================================================================
print("  -> Fig 5: Full comparison -- all 7 models")

all_model_colors = [BASELINE_COLOR]*len(baseline_names) + [ADVANCED_COLOR]*len(advanced_names)

r2f  = [avg(n, freq_cols, 'R2') * 100 for n in all_names]
r2b  = [avg(n, bw_cols,   'R2') * 100 for n in all_names]
maef = [avg(n, freq_cols, 'MAE')       for n in all_names]
maeb = [avg(n, bw_cols,   'MAE')       for n in all_names]

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
cmp_bar(axes[0,0], xall, r2f,  all_model_colors, all_abbrevs,
        '(a) Frequency R2 (%)  --  higher is better',
        'R2 Score (%)', ref=100, fmt='.2f')
cmp_bar(axes[0,1], xall, r2b,  all_model_colors, all_abbrevs,
        '(b) Bandwidth R2 (%)  --  higher is better',
        'R2 Score (%)', ref=100, fmt='.2f')
cmp_bar(axes[1,0], xall, maef, all_model_colors, all_abbrevs,
        '(c) Frequency MAE (GHz)  --  lower is better',
        'Mean Absolute Error (GHz)', fmt='.5f')
cmp_bar(axes[1,1], xall, maeb, all_model_colors, all_abbrevs,
        '(d) Bandwidth MAE (GHz)  --  lower is better',
        'Mean Absolute Error (GHz)', fmt='.5f')

fig.legend(handles=[Line2D([0],[0], color=c, marker='s', ls='', ms=10,
                    label=f'{ab} = {nm}')
                    for c, ab, nm in zip(ALL_PALETTE, all_abbrevs, all_names)],
           loc='lower center', ncol=4, fontsize=9.5,
           frameon=True, bbox_to_anchor=(0.5, -0.03))
fig.suptitle('Fig. 5 -- Full Comparison: All 7 Models\n'
             '(R2 Accuracy and MAE Error for Frequency and Bandwidth Prediction)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
save_show('plot_fig5_full_comparison.png')

# ==============================================================================
# FIG 6  -- Key comparison: ExtraTrees vs CatBoost vs Stacking_Ensemble
# ==============================================================================
print("  -> Fig 6: ExtraTrees vs CatBoost vs Stacking_Ensemble")

cmp_names   = ['Extra Trees', 'CatBoost', 'Stacking_Ensemble']
cmp_keys    = ['Extra Trees', 'CatBoost', 'Stacking_Ensemble']
cmp_colors  = [BASELINE_COLOR, ADVANCED_COLOR, ADVANCED_COLOR]

cmp_freq_r2 = [avg(k, freq_cols, 'R2') * 100 for k in cmp_keys]
cmp_bw_r2   = [avg(k, bw_cols,   'R2') * 100 for k in cmp_keys]
cmp_avg_r2  = [(f + b) / 2 for f, b in zip(cmp_freq_r2, cmp_bw_r2)]
cmp_freq_mae= [avg(k, freq_cols, 'MAE') for k in cmp_keys]
cmp_bw_mae  = [avg(k, bw_cols,   'MAE') for k in cmp_keys]

et_avg = cmp_avg_r2[0]
improvements = [0.0] + [v - et_avg for v in cmp_avg_r2[1:]]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# -- subplot 1: Freq R2 & BW R2 grouped bars
x3 = np.arange(3); w = 0.35
b1 = axes[0].bar(x3 - w/2, cmp_freq_r2, w, label='Freq R2 (%)',
                 color=['#1f78b4','#e07b54','#e07b54'], edgecolor='k')
b2 = axes[0].bar(x3 + w/2, cmp_bw_r2,   w, label='BW R2 (%)',
                 color=['#a6cee3','#fdbf6f','#fdbf6f'], edgecolor='k')
axes[0].set_xticks(x3); axes[0].set_xticklabels(cmp_names, fontsize=10)
axes[0].set_ylabel('R2 Score (%)', fontsize=11)
axes[0].set_ylim(88, 102)
axes[0].set_title('(a) Frequency & Bandwidth R2 (%)\n(higher is better)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].axhline(100, color='red', ls='--', lw=1, alpha=0.5)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2, h+0.05,
                     f'{h:.2f}%', ha='center', va='bottom',
                     fontsize=8.5, fontweight='bold')

# -- subplot 2: Average R2 with improvement annotation
bars2 = axes[1].bar(x3, cmp_avg_r2, color=cmp_colors, edgecolor='k', width=0.55)
axes[1].set_xticks(x3); axes[1].set_xticklabels(cmp_names, fontsize=10)
axes[1].set_ylabel('Average R2 (%)', fontsize=11)
axes[1].set_ylim(93, 102)
axes[1].set_title('(b) Average R2 (%) = (Freq R2 + BW R2) / 2\nImprovement over Extra Trees',
                  fontsize=11, fontweight='bold')
axes[1].axhline(et_avg, color='navy', ls='--', lw=1.5, alpha=0.7,
                label=f'Extra Trees baseline ({et_avg:.2f}%)')
axes[1].legend(fontsize=9)
for bar, v, imp in zip(bars2, cmp_avg_r2, improvements):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{v:.2f}%', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    if imp > 0:
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.35,
                     f'+{imp:.2f}% ✓', ha='center', va='bottom',
                     fontsize=9, color='green', fontweight='bold')

# -- subplot 3: MAE comparison (lower is better)
x3m = np.arange(3)
b3 = axes[2].bar(x3m - w/2, cmp_freq_mae, w, label='Freq MAE (GHz)',
                 color=['#1f78b4','#e07b54','#e07b54'], edgecolor='k', alpha=0.9)
b4 = axes[2].bar(x3m + w/2, cmp_bw_mae,   w, label='BW MAE (GHz)',
                 color=['#a6cee3','#fdbf6f','#fdbf6f'], edgecolor='k', alpha=0.9)
axes[2].set_xticks(x3m); axes[2].set_xticklabels(cmp_names, fontsize=10)
axes[2].set_ylabel('MAE (GHz)', fontsize=11)
axes[2].set_title('(c) Mean Absolute Error (GHz)\n(lower is better)',
                  fontsize=11, fontweight='bold')
axes[2].legend(fontsize=10)
for bars in [b3, b4]:
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x()+bar.get_width()/2, h+0.0003,
                     f'{h:.4f}', ha='center', va='bottom',
                     fontsize=8, rotation=90)

fig.legend(handles=[Line2D([0],[0], color=c, marker='s', ls='', ms=12,
                    label=nm)
                    for c, nm in zip(cmp_colors, cmp_names)],
           loc='lower center', ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Fig. 6 -- Key Comparison: ExtraTrees vs CatBoost vs Stacking_Ensemble\n'
             'Frequency R2, Bandwidth R2, and MAE comparison',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
save_show('plot_fig6_key_comparison.png')

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print(f"\n{'='*76}")
print("  RESULTS SUMMARY -- All 7 Models")
print(f"{'='*76}")
print(f"  {'Model':<24} {'Freq R2%':>8}  {'Freq MAE':>9}  "
      f"{'BW R2%':>8}  {'BW MAE':>9}")
print("  " + "-"*64)
for n in all_names:
    print(f"  {n:<24} "
          f"{avg(n,freq_cols,'R2')*100:>8.2f}%  {avg(n,freq_cols,'MAE'):>9.5f}  "
          f"{avg(n,bw_cols,'R2')*100:>8.2f}%  {avg(n,bw_cols,'MAE'):>9.5f}")
print(f"\n{'='*76}")
print("  ALL 6 FIGURES GENERATED  (displayed on screen + saved as PNG)")
print(f"{'='*76}")
print("""
  plot_fig1_histograms.png      -- Fig 1: Dataset distributions
  plot_fig2_error_accuracy.png  -- Fig 2: All 7 models error & accuracy
  plot_fig3_freq_scatter.png    -- Fig 3: All 7 models freq scatter
  plot_fig4_bw_scatter.png      -- Fig 4: All 7 models BW scatter
  plot_fig5_full_comparison.png -- Fig 5: All 7 models R2 & MAE comparison
  plot_fig6_key_comparison.png  -- Fig 6: ExtraTrees vs CatBoost vs Stacking
""")

