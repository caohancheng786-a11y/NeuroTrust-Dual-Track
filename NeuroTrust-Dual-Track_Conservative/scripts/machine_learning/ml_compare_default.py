# -*- coding: utf-8 -*-
"""对比：同样274全量数据，正式(嵌套CV选出的浅树强正则) vs XGB默认深树，看SHAP范围与排序。"""
import warnings;warnings.filterwarnings('ignore')
import pandas as pd,numpy as np
from xgboost import XGBRegressor
import shap
D=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
df=pd.read_csv(D+r'\建模数据集_量表分.csv')
Xcols=['PC','FS','PFE','WCE','PSC','PARV'];X=df[Xcols]
print('建模数据集样本量 N =',len(df),'（=全部问卷，无抽样、无删减）\n')
for t in['NT','ST']:
    y=df[t]
    m_formal=XGBRegressor(n_estimators=150,max_depth=1,learning_rate=.03,subsample=.8,
        colsample_bytree=.9,min_child_weight=5,reg_lambda=1.5,reg_alpha=.1,random_state=2026,n_jobs=1).fit(X,y)
    m_default=XGBRegressor(random_state=2026,n_jobs=1).fit(X,y)  # XGB默认：max_depth=6深树、弱正则
    print('========== 目标 %s =========='%t)
    for name,m in[('我们正式(CV选出:max_depth=1浅树+强正则)',m_formal),('VSCode那种(XGB默认:深树弱正则)',m_default)]:
        sv=np.asarray(shap.TreeExplainer(m).shap_values(X));ma=np.abs(sv).mean(0)
        order=list(np.argsort(-ma))
        rk=' > '.join('%s %.3f'%(Xcols[i],ma[i]) for i in order)
        print('  %s'%name)
        print('    SHAP值范围 [%.3f, %.3f]（范围越大/树越深→点铺得越开）'%(sv.min(),sv.max()))
        print('    贡献排序: %s\n'%rk)
