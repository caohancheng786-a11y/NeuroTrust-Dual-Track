# -*- coding: utf-8 -*-
"""补算 out-of-fold SHAP（老师新要求）：
每个外层折用训练集fit(内层5折选参) -> 只对该折【未见过的测试集】逐人算6维SHAP并保存；
50折(5x10)汇总：全部测试折预测=274人x10=2740条；同时给每人10次平均版(274行)。
折划分/超参网格/SEED与 ml_3_nested.full 完全一致，故每折|SHAP|均值必须=已归档逐折稳定性表(对拍校验)。"""
import json,warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from xgboost import XGBRegressor
import shap
SEED=2026;OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
cn={'PC':'感知控制','FS':'反馈稳定性','PFE':'感知反馈可解释性','WCE':'佩戴与校准体验','PSC':'隐私与监控担忧','PARV':'感知注意力调节价值'}
GRID={'max_depth':[1,2,3],'n_estimators':[150,300,500]}
data=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=data[Xcols].values.astype(float);n=len(data)
def xgb_fit(tr,ytr):
    base=XGBRegressor(learning_rate=.03,subsample=.8,colsample_bytree=.9,min_child_weight=5,
        reg_lambda=1.5,reg_alpha=.1,random_state=SEED,n_jobs=1,verbosity=0,objective='reg:squarederror')
    gs=GridSearchCV(base,GRID,cv=5,scoring='neg_mean_squared_error',n_jobs=1);gs.fit(X[tr],ytr)
    return gs.best_estimator_
# 已归档逐折稳定性表，用于对拍（折内|SHAP|均值必须一致）
fold_disk=pd.read_csv(OUT+r'\shap_fold_stability_full.csv')
summary={}
for tgt in['NT','ST']:
    y=data[tgt].values.astype(float);rows=[];check_maxdiff=0
    rkf=RepeatedKFold(n_splits=5,n_repeats=10,random_state=SEED)
    for k,(tr,te) in enumerate(rkf.split(X)):
        xg=xgb_fit(tr,y[tr]);sv=np.asarray(shap.TreeExplainer(xg).shap_values(X[te]))
        # 对拍：本折测试集|SHAP|均值 vs 归档逐折表
        disk=fold_disk[(fold_disk.target==tgt)&(fold_disk.fold==k)].set_index('feat').loc[Xcols]['mean_abs'].values
        check_maxdiff=max(check_maxdiff,float(np.abs(np.abs(sv).mean(0)-disk).max()))
        for ii,i in enumerate(te):
            r=dict(idx=int(i),repeat=k//5,fold=k)
            for j,f in enumerate(Xcols):r[f+'_shap']=float(sv[ii,j]);r[f+'_x']=float(X[i,j])
            rows.append(r)
    df=pd.DataFrame(rows).sort_values(['idx','fold']).reset_index(drop=True)
    df.to_csv(OUT+r'\shap_oof_%s.csv'%tgt,index=False,encoding='utf-8-sig')
    scol=[f+'_shap' for f in Xcols]
    # 全部2740条
    meanabs=df[scol].abs().mean();signed=df[scol].mean()
    # 每人10折平均 -> 274行
    per=df.groupby('idx')[scol].mean()
    per.columns=Xcols;per.to_csv(OUT+r'\shap_oof_%s_每人平均.csv'%tgt,encoding='utf-8-sig')
    order=list(meanabs.sort_values(ascending=False).index.str.replace('_shap',''))
    summary[tgt]=dict(n_test_predictions=len(df),n_persons=per.shape[0],
        mean_abs={f:round(float(meanabs[f+'_shap']),4) for f in Xcols},
        signed_mean={f:round(float(signed[f+'_shap']),4) for f in Xcols},
        per_person_mean_abs={f:round(float(per[f].abs().mean()),4) for f in Xcols},rank=order)
    print('===== %s ====='%tgt)
    print('OOF测试预测条数=%d（=274x10），对拍归档逐折|SHAP|最大差=%.2e（应≈0）'%(len(df),check_maxdiff))
    print('OOF Mean|SHAP|排序:', ' > '.join('%s %.4f'%(f,meanabs[f+'_shap']) for f in order))
    print('OOF带符号均值(方向):',{f:round(float(signed[f+'_shap']),4) for f in Xcols})
json.dump(summary,open(OUT+r'\shap_oof_summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('\n[saved] shap_oof_NT/ST.csv + 每人平均.csv + shap_oof_summary.json')
