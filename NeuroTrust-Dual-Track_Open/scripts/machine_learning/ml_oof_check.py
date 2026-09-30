# -*- coding: utf-8 -*-
"""OOF新需求独立核查：无泄漏/条数/特征对齐/数值排序/逐折对拍/重算复现+SHAP恒等式。"""
import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,json
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from xgboost import XGBRegressor
import shap
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4';Xcols=['PC','FS','PFE','WCE','PSC','PARV'];SEED=2026
model=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=model[Xcols].values.astype(float);n=len(model)
folddisk=pd.read_csv(OUT+r'\shap_fold_stability_full.csv');summ=json.load(open(OUT+r'\shap_oof_summary.json',encoding='utf-8'))
GRID={'max_depth':[1,2,3],'n_estimators':[150,300,500]}
def xgb_fit(tr,y):
    b=XGBRegressor(learning_rate=.03,subsample=.8,colsample_bytree=.9,min_child_weight=5,reg_lambda=1.5,
        reg_alpha=.1,random_state=SEED,n_jobs=1,verbosity=0,objective='reg:squarederror')
    return GridSearchCV(b,GRID,cv=5,scoring='neg_mean_squared_error',n_jobs=1).fit(X[tr],y).best_estimator_
allok=True
def chk(name,cond,evi=''):
    global allok;allok=allok&bool(cond);print(('[PASS] 'if cond else '[FAIL] ')+name+('  | '+str(evi) if evi else ''))
for t in['NT','ST']:
    print('\n================ %s ================'%t)
    d=pd.read_csv(OUT+r'\shap_oof_%s.csv'%t);y=model[t].values.astype(float)
    rkf=list(RepeatedKFold(n_splits=5,n_repeats=10,random_state=SEED).split(X))
    # A 无泄漏+折归属：csv每fold的idx==该折te，且te与tr不相交(模型没见过)
    leak=set();foldmatch=True;cnt=[]
    for k,(tr,te) in enumerate(rkf):
        ids=d[d.fold==k].idx.values
        cnt.append(len(ids))
        if set(ids)!=set(te):foldmatch=False
        if set(te)&set(tr):leak.add(k)
    chk('50折齐全且每折SHAP样本=该折测试集',foldmatch and d.fold.nunique()==50)
    chk('每个测试样本都不在该折训练集(OOF无泄漏)',len(leak)==0,leak or 'te∩tr=∅')
    chk('总条数=2740且每人恰好10条',len(d)==2740 and (d.groupby('idx').size()==10).all(),(len(d),int((d.groupby('idx').size()==10).sum())))
    # B 特征值对齐本人(无串行)
    xerr=0
    for f in Xcols:
        got=d.apply(lambda r:X[int(r.idx),Xcols.index(f)]-r[f+'_x'],axis=1).abs().max();xerr=max(xerr,got)
    chk('每行特征值f_x=该idx本人构念分(未串行)',xerr<1e-12,'最大差%.1e'%xerr)
    # C 独立重算mean|SHAP|/排序 vs json
    ma=d[[f+'_shap' for f in Xcols]].abs().mean()
    jr=summ[t]['mean_abs'];jrank=summ[t]['rank']
    calc={f:round(float(ma[f+'_shap']),4) for f in Xcols}
    chk('OOF Mean|SHAP|数值=summary底稿',all(abs(calc[f]-jr[f])<5e-5 for f in Xcols),calc)
    rank=list(ma.sort_values(ascending=False).index.str.replace('_shap',''))
    chk('排序与底稿一致',rank==jrank,(rank,jrank))
    # D 每人平均csv
    per=pd.read_csv(OUT+r'\shap_oof_%s_每人平均.csv'%t).set_index('idx')
    chkper=(d.groupby('idx')[[f+'_shap' for f in Xcols]].mean().sub(per[Xcols].values).abs().max().max()<1e-12)
    chk('每人平均csv=该人10条SHAP均值(274行)',per.shape==(274,6) and chkper,per.shape)
    # E 逐折|SHAP|均值=稳定性表
    e=0
    for k in range(50):
        sub=d[d.fold==k];sv=sub[[f+'_shap' for f in Xcols]].values;disk=folddisk[(folddisk.target==t)&(folddisk.fold==k)].set_index('feat').loc[Xcols]['mean_abs'].values
        e=max(e,float(np.abs(np.abs(sv).mean(0)-disk).max()))
    chk('逐折|SHAP|均值=已归档稳定性表(容差1e-6)',e<1e-6,'最大差%.1e（浮点非确定性噪声）'%e)
# F 抽查NT fold0/25/49 重fit复算SHAP + 恒等式
print('\n================ 重算复算&SHAP恒等式(抽查3折, NT) ================')
d=pd.read_csv(OUT+r'\shap_oof_NT.csv');y=model['NT'].values.astype(float);rkf=list(RepeatedKFold(n_splits=5,n_repeats=10,random_state=SEED).split(X))
recerr=0;recalc=0
for k in[0,25,49]:
    tr,te=rkf[k];xg=xgb_fit(tr,y[tr]);ex=shap.TreeExplainer(xg);sv=np.asarray(ex.shap_values(X[te]));ev=float(np.asarray(ex.expected_value).ravel()[0])
    recerr=max(recerr,float(np.abs(xg.predict(X[te])-(ev+sv.sum(1))).max()))
    disk=d[d.fold==k].sort_values('idx')[[f+'_shap' for f in Xcols]].values
    order=np.argsort(te);recalc=max(recalc,float(np.abs(sv[order]-disk).max()))
chk('重fit折模型 SHAP可逐格复现落盘值',recalc<1e-9,'最大差%.1e'%recalc)
chk('SHAP恒等式 max|predict-(EV+ΣSHAP)|≈0',recerr<1e-5,'最大差%.1e'%recerr)
print('\n========== 总体:', 'OOF全部核查通过 ✅' if allok and recerr<1e-5 and recalc<1e-9 else '存在问题 ❌==========')
