# -*- coding: utf-8 -*-
"""Figure 3（修订）：英文彩色 OOF beeswarm，NT/ST 双面板（手绘，等价shap beeswarm）。"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

OUT = r'F:\研究生gpt\周老师\9.9处理\2\3\4'
SAVE = r'F:\研究生gpt\周老师\MDPI审稿意见\Figure3_Revised_OOF_beeswarm_NT_ST_dual'
Xcols = ['PC','FS','PFE','WCE','PSC','PARV']
en = {'PC':'Perceived Control (PC)', 'FS':'Feedback Stability (FS)',
      'PFE':'Perceived Feedback Explainability (PFE)',
      'WCE':'Wear & Calibration Experience (WCE)',
      'PSC':'Privacy & Surveillance Concern (PSC)',
      'PARV':'Perceived Attention Regulation Value (PARV)'}
from matplotlib.colors import ListedColormap
import os as _os
_cmap_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'ref_cmap.npy')
cmap = ListedColormap(np.load(_cmap_file))

oof = {t: pd.read_csv(OUT + r'\shap_oof_%s.csv' % t) for t in ['NT','ST']}

def beeswarm(ax, sv, xx, names, seed):
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)
    rng = np.random.default_rng(seed)
    for row, j in enumerate(order):
        s, xv = sv[:, j], xx[:, j]
        span = np.nanmax(xv) - np.nanmin(xv)
        xn = (xv - np.nanmin(xv)) / (span + 1e-9)
        idx = np.argsort(s)
        yj = row + rng.normal(0, 0.08, len(s))   # 与 shap summary_plot 一致的紧凑垂直抖动
        ax.scatter(s[idx], yj[idx], c=xn[idx], cmap=cmap, s=9,
                   alpha=0.82, edgecolors='none', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([names[j] for j in order], fontsize=11)
    ax.axvline(0, color='#999999', lw=0.8)
    ax.set_ylim(-0.6, 5.6)
    ax.set_xlabel('SHAP value (out-of-fold impact)', fontsize=12)
    ax.grid(axis='x', alpha=0.15)

fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.4))
fig.subplots_adjust(left=0.20, right=0.71, wspace=0.28, top=0.86, bottom=0.12)
for k, (ax, t) in enumerate(zip(axes, ['NT','ST'])):
    d = oof[t]
    sv = d[[f+'_shap' for f in Xcols]].values
    xx = d[[f+'_x' for f in Xcols]].values
    beeswarm(ax, sv, xx, [en[f] for f in Xcols], seed=100+k)
    ax.set_title('(%s) %s' % ('a' if k==0 else 'b',
                 'Neuro-Trust (NT)' if t=='NT' else 'System Trust (ST)'),
                 fontsize=14, fontweight='bold')

# 右面板标签移到右侧，避免伸入左面板
axes[1].yaxis.tick_right()

cax = fig.add_axes([0.905, 0.12, 0.016, 0.74])
cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1)), cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'])
cbar.set_label('Feature value', fontsize=11)

fig.suptitle('Out-of-fold SHAP direction, aggregated across held-out outer test folds; '
             'red = high feature value, points to the right raise predicted trust',
             fontsize=12, y=0.96)
fig.savefig(SAVE + '.png', dpi=600, facecolor='white')
fig.savefig(SAVE + '.pdf', facecolor='white')
print("saved", SAVE)
