# -*- coding: utf-8 -*-
"""独立交叉验证（与主脚本不同实现路径）：数据正确性 + SHAP数学恒等式 + 多方法排序稳健性。"""
import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import shap,json
D=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
SRC=r'F:\研究生gpt\周老师\9.9处理\2\0909-按分数_青年脑机接口注意力训练体验与信任调查.xlsx'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
# 独立重写分组（便于肉眼核对题项归属）
G={'NT':['Q10','Q11','Q15','Q16'],'ST':['Q12','Q14','Q20','Q22'],'PC':['Q13','Q19','Q23','Q27'],
'FS':['Q9','Q29','Q35','Q40'],'PFE':['Q17','Q18','Q21','Q36'],'WCE':['Q6','Q7','Q8','Q34'],
'PSC':['Q24','Q25','Q26','Q28'],'PARV':['Q30','Q31','Q32','Q33']}
raw=pd.read_excel(SRC)
df=pd.DataFrame({g:raw[qs].apply(pd.to_numeric).mean(axis=1) for g,qs in G.items()})
print('【1】题项分组（请肉眼核对，共36题无重复）:')
allq=[q for v in G.values() for q in v];print('  题项总数',len(allq),'去重后',len(set(allq)),'应为36且无重复:',len(set(allq))==36)
for g,v in G.items():print('   %-4s <- %s'%(g,v))
X=df[Xcols];print('\n【2】列名顺序（确认建模X未错位）:',list(X.columns))
# 相关：确认NT/ST没接反
c=df[Xcols+['NT','ST']].corr()
print('\n【3】各前因与NT/ST的简单相关（PARV应明显偏NT、PFE应偏ST）:')
print(c.loc[Xcols,['NT','ST']].round(3).to_string())
print('  PARV: r(NT)=%.3f vs r(ST)=%.3f  -> %s'%(c.loc['PARV','NT'],c.loc['PARV','ST'],'偏NT ✓' if c.loc['PARV','NT']>c.loc['PARV','ST'] else '异常!'))
print('  PFE : r(NT)=%.3f vs r(ST)=%.3f  -> %s'%(c.loc['PFE','NT'],c.loc['PFE','ST'],'偏ST ✓' if c.loc['PFE','ST']>c.loc['PFE','NT'] else '异常!'))
# 与落盘建模csv逐格比对
disk=pd.read_csv(D+r'\建模数据集_量表分.csv')
same=all(np.allclose(df[g],disk[g]) for g in G)
print('\n【4】独立重算构念 vs 落盘csv 逐格一致:',same)

def ranks_from_imp(imp):return list(pd.Series(imp,index=Xcols).rank(ascending=False,method='min').astype(int))
summary={}
for tgt in['NT','ST']:
    y=df[tgt].values;Xs=pd.DataFrame(StandardScaler().fit_transform(X),columns=Xcols);Z=Xs.apply(zscore).values
    print('\n================ 目标 %s ================'%tgt)
    method_rank={}
    # (a) OLS 标准化回归（最传统方法，独立于树模型）
    m=sm.OLS(y,sm.add_constant(Z)).fit()
    beta=pd.Series(m.params[1:],index=Xcols);pval=pd.Series(m.pvalues[1:],index=Xcols)
    method_rank['OLS标准化β']=ranks_from_imp(beta.abs())
    print('【5】OLS多元标准化回归系数(方向/大小/p):')
    for f in Xcols:print('   %-4s β=%+.3f p=%.3f'%(f,beta[f],pval[f]))
    print('   OLS样本内R²=%.3f'%m.rsquared)
    # (b) ElasticNet 独立重算
    en=ElasticNetCV(l1_ratio=[.1,.5,.9,1],cv=5,max_iter=80000).fit(Xs,y)
    method_rank['ElasticNet']=ranks_from_imp(np.abs(en.coef_))
    # (c)(d)(e) XGB不同深度/种子
    xgb_runs={}
    for tag,dep,sd in[('XGB_d1_s2026',1,2026),('XGB_d3_s0',3,0),('XGB_d2_s999',2,999)]:
        xb=XGBRegressor(n_estimators=300,max_depth=dep,learning_rate=.03,subsample=.8,colsample_bytree=.9,
            min_child_weight=5,reg_lambda=1.5,reg_alpha=.1,random_state=sd,n_jobs=1,verbosity=0).fit(X,y)
        ex=shap.TreeExplainer(xb);sv=np.asarray(ex.shap_values(X))
        xgb_runs[tag]=(xb,ex,sv)
        method_rank[tag]=ranks_from_imp(np.abs(sv).mean(0))
    # SHAP数学恒等式 local accuracy: ev + sum(shap) == predict
    xb0,ex0,sv0=xgb_runs['XGB_d1_s2026'];pred=xb0.predict(X);recon=ex0.expected_value+sv0.sum(1)
    err=np.abs(pred-recon).max()
    print('【6】SHAP恒等式检验 max|predict-(EV+ΣSHAP)|=%.2e（应≈0，证明SHAP值本身算对）'%err)
    # XGB原生gain重要性
    method_rank['XGB原生gain']=ranks_from_imp(xb0.feature_importances_)
    # (f) 随机森林 impurity
    rf=RandomForestRegressor(n_estimators=500,max_depth=4,random_state=2026,n_jobs=1).fit(X,y)
    method_rank['随机森林impurity']=ranks_from_imp(rf.feature_importances_)
    # (g) 随机森林 permutation importance（独立于树内部打分）
    perm=permutation_importance(rf,X,y,n_repeats=20,random_state=2026,n_jobs=1)
    method_rank['随机森林置换重要性']=ranks_from_imp(np.abs(perm.importances_mean))
    # 训练R² vs 正式OOF R²
    rj=json.load(open(D+r'\ml_nested_results_full.json',encoding='utf-8'))['targets'][tgt]['perf']
    print('【7】全量训练R²(XGB)=%.3f  vs 正式嵌套CV的OOF R²: EN %.3f / XGB %.3f（CV更低=无泄漏、正常）'%(
        xb0.score(X,y),rj['EN']['R2'],rj['XGB']['R2']))
    tab=pd.DataFrame(method_rank,index=Xcols)
    tab['平均排名']=tab.mean(1).round(2);tab=tab.sort_values('平均排名')
    print('【8】七种独立方法的特征排名（1=最重要，跨方法一致才稳）:')
    print(tab.to_string())
    summary[tgt]=tab
print('\n================ 跨方法稳健性总结 ================')
for tgt in['NT','ST']:
    t=summary[tgt]
    print('%s 各方法第1名出现情况:'%tgt,dict(t[[c for c in t.columns if c!="平均排名"]].apply(lambda r:(r==1).sum())))
    print(tgt,'综合排名(平均):',' > '.join(t.index.tolist()))
