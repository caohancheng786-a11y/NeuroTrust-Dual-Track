# -*- coding: utf-8 -*-
"""重画 Figure 1（核心SEM路径），补 WCE->NT (β=.154*)。600 dpi PNG + vector PDF。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

fig, ax = plt.subplots(figsize=(13.2, 7.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

C_ANT = dict(facecolor='#F2F2F2', edgecolor='#7F7F7F')
C_NT = dict(facecolor='#DCE9F7', edgecolor='#8FB3D9')
C_ST = dict(facecolor='#E3F0E3', edgecolor='#8FBC8F')
C_RI = dict(facecolor='#FFFFFF', edgecolor='#7F7F7F')

def box(x, y, w, h, colors, lines, fs=12.5, title_fs=13.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.6",
                       linewidth=1.4, **colors)
    ax.add_patch(p)
    cx, cy = x + w/2, y + h/2
    n = len(lines)
    step = h/(n+0.4)
    for i, (txt, bold, fs_) in enumerate(lines):
        yy = cy + step*(n-1)/2 - i*step
        ax.text(cx, yy, txt, ha='center', va='center', fontsize=fs_,
                fontweight='bold' if bold else 'normal')

# 前因（左列，4个）
box(2, 76, 26, 12, C_ANT, [("PARV", True, 13.5), ("Attention-regulation value", False, 11.5)])
box(2, 58, 26, 12, C_ANT, [("WCE", True, 13.5), ("Wear & calibration experience", False, 11.5)])
box(2, 40, 26, 12, C_ANT, [("PFE", True, 13.5), ("Feedback explainability", False, 11.5)])
box(2, 18, 26, 12, C_ANT, [("PSC", True, 13.5), ("Privacy concern", False, 11.5)])

# NT / ST / RI
box(45, 60, 22, 18, C_NT, [("Neuro-Trust", True, 14), ("(NT)", True, 13),
                           (r"$R^2=0.358$", False, 12.5)])
box(45, 24, 22, 18, C_ST, [("System Trust", True, 14), ("(ST)", True, 13),
                           (r"$R^2=0.562$", False, 12.5)])
box(78, 40, 20, 20, C_RI, [("Reuse Intention", True, 13.5), ("(RI)", True, 13),
                           (r"$R^2=0.271$", False, 12.5)])

def arrow(p0, p1, label, lpos=None, fs=12):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color='#222222',
                                shrinkA=0, shrinkB=0))
    if label:
        lx, ly = lpos if lpos else ((p0[0]+p1[0])/2, (p0[1]+p1[1])/2)
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.2, alpha=0.9))

# 前因 -> NT/ST
arrow((28, 84), (45, 74), r"β=.380***", (36.5, 81))
arrow((28, 66), (45, 68), r"β=.154*", (36.5, 69.5))
arrow((28, 48), (45, 64), r"β=.346***", (36.5, 58.5))
arrow((28, 46), (45, 40), r"β=.538***", (36.5, 43))
arrow((28, 24), (45, 32), r"β=−.198*", (36.5, 26))
# NT/ST -> RI
arrow((67, 72), (78, 56), r"β=.359***", (72.5, 67))
arrow((67, 34), (78, 46), r"β=.255***", (72.5, 38))
# NT <-> ST 相关（双向）
ax.annotate("", xy=(56, 60), xytext=(56, 42),
            arrowprops=dict(arrowstyle="<->", lw=1.4, color='#555555'))
ax.text(60, 51, "HTMT=.411\nSeparate CFA > merged CFA", ha='left', va='center',
        fontsize=11, color='#333333')

ax.text(50, 96.5, "Core empirical pattern: shared antecedent + track-specific antecedent",
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(50, 6, "All six candidate antecedents were entered simultaneously; only statistically "
        "supported paths are shown.", ha='center', va='center', fontsize=11.5, style='italic')

OUT = r'F:\研究生gpt\周老师\MDPI审稿意见\Figure1_Revised_SEM_core_model'
fig.savefig(OUT + '.png', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '.pdf', bbox_inches='tight', facecolor='white')
print("saved", OUT)
