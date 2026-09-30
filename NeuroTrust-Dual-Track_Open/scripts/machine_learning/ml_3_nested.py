# -*- coding: utf-8 -*-
"""
机器学习主分析（严格按老师11步规范）
- 两平行任务: 6前因 -> NT / ST
- Repeated Nested 5-fold CV: 外层5折x10重复=50测试折; 内层5折为 ElasticNet / XGBoost 选超参
- 报告 OOF R2/RMSE/MAE; 每外层折存SHAP -> 稳定性(平均排名/进Top2比例); 全量模型出最终SHAP与方向
用法: python ml_3_nested.py quick | full
"""
import sys,json,warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
import shap
MODE=sys.argv[1] if len(sys.argv)>1 else 'quick';FULL=(MODE=='full')
REPS=10 if FULL else 1
GRID=({'max_depth':[1,2,3],'n_estimators':[150,300,500]} if FULL
      else {'max_depth':[1,2],'n_estimators':[200]})
SEED=2026;OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
cn={'PC':'感知控制','FS':'反馈稳定性','PFE':'感知反馈可解释性','WCE':'佩戴与校准体验','PSC':'隐私与监控担忧','PARV':'感知注意力调节价值'}
data=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=data[Xcols].values.astype(float);n=len(data);p=6
def en_fit(tr,ytr):
    m=Pipeline([('sc',StandardScaler()),('en',ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,1.],cv=5,
        max_iter=80000,random_state=SEED,selection='cyclic'))]);return m.fit(X[tr],ytr)
def xgb_fit(tr,ytr,tune=True):
    base=XGBRegressor(learning_rate=.03,subsample=.8,colsample_bytree=.9,min_child_weight=5,
        reg_lambda=1.5,reg_alpha=.1,random_state=SEED,n_jobs=1,verbosity=0,objective='reg:squarederror')
    if tune:
        gs=GridSearchCV(base,GRID,cv=5,scoring='neg_mean_squared_error',n_jobs=1)
        gs.fit(X[tr],ytr);return gs.best_estimator_,gs.best_params_
    base.fit(X[tr],ytr);return base,{}
results={'mode':MODE,'N':n,'outer':'5fold x %d repeats = %d test folds'%(REPS,5*REPS),
         'inner':'5fold tuning for both EN(l1/alpha) and XGBoost(depth/nest)','features':Xcols,'targets':{}}
fold_rows=[]
for tgt in['NT','ST']:
    y=data[tgt].values.astype(float)
    rkf=RepeatedKFold(n_splits=5,n_repeats=REPS,random_state=SEED)
    oof={'EN':[[] for _ in range(n)],'XGB':[[] for _ in range(n)]}
    fold_rmse={'EN':[],'XGB':[]};chosen=[];fold_shap=[]
    for k,(tr,te) in enumerate(rkf.split(X)):
        en=en_fit(tr,y[tr]);pe=en.predict(X[te])
        xg,bp=xgb_fit(tr,y[tr]);px=xg.predict(X[te]);chosen.append(bp)
        fold_rmse['EN'].append(mean_squared_error(y[te],pe)**.5);fold_rmse['XGB'].append(mean_squared_error(y[te],px)**.5)
        for i,v in zip(te,pe):oof['EN'][i].append(v)
        for i,v in zip(te,px):oof['XGB'][i].append(v)
        # 该折测试集SHAP（折内模型），用于稳定性
        sv=np.abs(np.asarray(shap.TreeExplainer(xg).shap_values(X[te]))).mean(0)
        rk=pd.Series(sv,index=Xcols).rank(ascending=False,method='min').astype(int)
        share=sv/sv.sum()
        for j,f in enumerate(Xcols):
            fold_rows.append(dict(target=tgt,fold=k,feat=f,mean_abs=float(sv[j]),share=float(share[j]),rank=int(rk[f])))
    perf={}
    for m_ in('EN','XGB'):
        pred=np.array([np.mean(oof[m_][i]) for i in range(n)])
        perf[m_]=dict(R2=round(float(r2_score(y,pred)),4),RMSE=round(float(mean_squared_error(y,pred)**.5),4),
            MAE=round(float(mean_absolute_error(y,pred)),4),
            foldRMSE_mean=round(float(np.mean(fold_rmse[m_])),4),foldRMSE_std=round(float(np.std(fold_rmse[m_])),4))
    # SHAP稳定性
    fs=pd.DataFrame([r for r in fold_rows if r['target']==tgt])
    stab=[]
    for f in Xcols:
        d=fs[fs.feat==f]
        stab.append(dict(feat=f,cn=cn[f],mean_abs=round(d.mean_abs.mean(),4),mean_abs_sd=round(d.mean_abs.std(),4),
            share_pct=round(100*d.share.mean(),1),mean_rank=round(d['rank'].mean(),2),
            top2_pct=round(100*(d['rank']<=2).mean(),0),top3_pct=round(100*(d['rank']<=3).mean(),0)))
    stab=sorted(stab,key=lambda z:-z['mean_abs'])
    # 全量最终模型
    enf=en_fit(np.arange(n),y);encoef=enf.named_steps['en'].coef_
    xgf,bpf=xgb_fit(np.arange(n),y);expl=shap.TreeExplainer(xgf);svf=np.asarray(expl.shap_values(X));ev=float(np.asarray(expl.expected_value).ravel()[0])
    mabs=np.abs(svf).mean(0);msigned=svf.mean(0);order=np.argsort(-mabs)
    finalimp=[dict(feat=Xcols[j],cn=cn[Xcols[j]],mean_abs_shap=round(float(mabs[j]),4),
                  signed_shap=round(float(msigned[j]),4),en_coef_std=round(float(encoef[j]),4),
                  fullrank=int(list(order).index(j))+1) for j in order]
    # 选中超参众数
    import collections
    md=collections.Counter(c.get('max_depth') for c in chosen if c).most_common(1)[0][0]
    ne=collections.Counter(c.get('n_estimators') for c in chosen if c).most_common(1)[0][0]
    results['targets'][tgt]=dict(perf=perf,stability=stab,final_importance=finalimp,
        full_best=dict(max_depth=md,n_estimators=ne),shap_expected=ev)
    if FULL:pd.DataFrame(svf,columns=Xcols).assign(target=tgt).to_csv(OUT+rf'\shap_full_{tgt}.csv',index=False,encoding='utf-8-sig')
    print('\n===== %s ====='%tgt);print('OOF  EN ',perf['EN']);print('OOF XGB',perf['XGB'])
    print('外层常选XGB超参 max_depth=%s n_est=%s'%(md,ne))
    print('SHAP稳定性(按平均贡献): 特征 平均|SHAP| 份额% 平均排名 进Top2比例')
    for s in stab:print('  %-4s %-8s %.4f %5.1f%% 排名%.2f Top2=%d/%-d'%(s['feat'],s['cn'],s['mean_abs'],s['share_pct'],s['mean_rank'],int(s['top2_pct']/100*5*REPS),5*REPS))
    print('全量SHAP排序/方向/EN系数:')
    for d in finalimp:print('  #%d %-4s |SHAP|=%.4f 方向=%.4f EN=%.3f'%(d['fullrank'],d['feat'],d['mean_abs_shap'],d['signed_shap'],d['en_coef_std']))
pd.DataFrame(fold_rows).to_csv(OUT+r'\shap_fold_stability_%s.csv'%MODE,index=False,encoding='utf-8-sig')
json.dump(results,open(OUT+r'\ml_nested_results_%s.json'%MODE,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('\n[saved] ml_nested_results_%s.json + shap_fold_stability_%s.csv'%(MODE,MODE))
