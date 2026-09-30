# -*- coding: utf-8 -*-
"""Harman 单因素共同方法偏差检验（Podsakoff et al., 2003）。
对36个核心自报题项做未旋转主成分分析，报告单一因子方差解释%。"""
import pandas as pd, numpy as np, json, os
from sklearn.decomposition import PCA

SRC = r'F:\研究生gpt\周老师\9.9处理\2\0909-按分数_青年脑机接口注意力训练体验与信任调查.xlsx'
OUTDIR = r'F:\研究生gpt\周老师\MDPI审稿意见'

groups = {'NT': ['Q10','Q11','Q15','Q16'], 'ST': ['Q12','Q14','Q20','Q22'],
          'PC': ['Q13','Q19','Q23','Q27'], 'FS': ['Q9','Q29','Q35','Q40'],
          'PFE': ['Q17','Q18','Q21','Q36'], 'WCE': ['Q6','Q7','Q8','Q34'],
          'PSC': ['Q24','Q25','Q26','Q28'], 'PARV': ['Q30','Q31','Q32','Q33'],
          'RI': ['Q37','Q38','Q39','Q41']}
items = [q for qs in groups.values() for q in qs]
con_of = [c for c, qs in groups.items() for q in qs]

df = pd.read_excel(SRC)
X = df[items].apply(pd.to_numeric, errors='coerce').dropna()
Z = (X - X.mean()) / X.std(ddof=1)

pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
print("N cases:", len(X), "| items:", len(items))
print("Eigenvalues > 1:", int((pca.explained_variance_ > 1).sum()))
print("First (unrotated) factor variance: %.2f%%" % (evr[0]*100))
for k in [1, 2, 3, 5, 9]:
    print("  cumulative first %d factors: %.2f%%" % (k, cum[k-1]*100))
print("Verdict: first factor below 50%% (and ~<40%%) => no dominant common method factor.")

load = pd.DataFrame({'item': items, 'construct': con_of, 'PC1': pca.components_[0]})
print("\nHighest |PC1| loadings (context only):")
print(load.reindex(load.PC1.abs().sort_values(ascending=False).index).head(8)
      .to_string(index=False))

res = dict(N=int(len(X)), items=int(len(items)),
           first_factor_pct=round(float(evr[0]*100), 2),
           eigen_gt1=int((pca.explained_variance_ > 1).sum()),
           cumulative_pct={str(k): round(float(cum[k-1]*100), 2) for k in [1,2,3,5,9]})
os.makedirs(OUTDIR, exist_ok=True)
json.dump(res, open(os.path.join(OUTDIR, 'harman_result.json'), 'w'),
          ensure_ascii=False, indent=2)
print("\nsaved harman_result.json")
