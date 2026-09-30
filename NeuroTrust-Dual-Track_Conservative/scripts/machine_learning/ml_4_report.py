# -*- coding: utf-8 -*-
"""读取full嵌套CV结果，生成：性能表/SHAP稳定性/SHAP×SEM整合/Q42三方对照 的Excel + 图。"""
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
import shap
from matplotlib.colors import LinearSegmentedColormap
# 科研灰度配色：浅灰(低)->黑(高)；条形主色黑，需区分系列时用浅灰
gray_cmap=LinearSegmentedColormap.from_list('shapgray',['#ececec','#a0a0a0','#000000'])
BLACK='#262626';GRAY_L='#bdbdbd';GRAY_M='#7d7d7d'
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
cn={'PC':'感知控制','FS':'反馈稳定性','PFE':'感知反馈可解释性','WCE':'佩戴与校准体验','PSC':'隐私与监控担忧','PARV':'感知注意力调节价值'}
R=json.load(open(OUT+r'\ml_nested_results_full.json',encoding='utf-8'))
data=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=data[Xcols].values.astype(float)
q42=pd.read_csv(OUT+r'\Q42_主观优先级.csv')
# 冻结SEM标准化β与显著性（模型B）
sem={  # (beta_NT, sig_NT, beta_ST, sig_ST)
'PC':(-.020,'ns',.155,'ns'),'FS':(-.056,'ns',.176,'边缘'),'PFE':(.346,'***',.538,'***'),
'WCE':(.154,'*',.130,'边缘'),'PSC':(-.024,'ns',-.198,'*'),'PARV':(.380,'***',-.008,'ns')}
NFOLD=50
# ---------- ① 性能 ----------
perf=[]
for t in['NT','ST']:
    for m,mn in[('EN','Elastic Net(线性基准)'),('XGB','XGBoost(主模型)')]:
        d=R['targets'][t]['perf'][m]
        perf.append(dict(预测目标=t,模型=mn,R2_CV=d['R2'],RMSE=d['RMSE'],MAE=d['MAE'],
                         折RMSE均值=d['foldRMSE_mean'],折RMSE标准差=d['foldRMSE_std']))
perf=pd.DataFrame(perf)
# ---------- ②③ SHAP稳定性+全量 ----------
def stab_df(t):
    s=pd.DataFrame(R['targets'][t]['stability']);s=s.rename(columns={'feat':'代码','cn':'因素',
        'mean_abs':'平均SHAP','mean_abs_sd':'SHAP标准差','share_pct':'贡献份额%','mean_rank':'平均排名',
        'top2_pct':'进Top2比例','top3_pct':'进Top3比例'})
    s['进Top2次数']=(s['进Top2比例']/100*NFOLD).round().astype(int)
    fin={d['feat']:d for d in R['targets'][t]['final_importance']}
    s['全量SHAP排名']=s['代码'].map(lambda f:fin[f]['fullrank'])
    s['全量平均SHAP']=s['代码'].map(lambda f:fin[f]['mean_abs_shap'])
    s['SHAP方向(带符号均值)']=s['代码'].map(lambda f:fin[f]['signed_shap'])
    s['ElasticNet标准化系数']=s['代码'].map(lambda f:fin[f]['en_coef_std'])
    return s[['代码','因素','全量SHAP排名','全量平均SHAP','SHAP方向(带符号均值)','平均排名','进Top2次数','进Top2比例','进Top3比例','贡献份额%','平均SHAP','SHAP标准差','ElasticNet标准化系数']]
sNT=stab_df('NT');sST=stab_df('ST')
# ---------- ④ SHAP × SEM 整合 ----------
ntrank={r['代码']:r['全量SHAP排名'] for _,r in sNT.iterrows()}
strank={r['代码']:r['全量SHAP排名'] for _,r in sST.iterrows()}
def role(f):
    a,b=ntrank[f],strank[f]
    if a<=2 and b>=4:return'NT偏向'
    if b<=2 and a>=4:return'ST偏向'
    if a<=3 and b<=3:return'双轨共享'
    return'次要'
concl={'PARV':'NT特异前因：SEM β→NT=.380***/→ST=ns，且ML NT第1(48/50进Top2)、ST第5(0/50)，结构特异+预测特异双重证据',
'PFE':'双轨共享前因：SEM两任务均强显著，ML NT第2/ST第1(50/50)，两条轨道都靠它',
'FS':'偏ST的次要前因：SEM仅ST边缘显著，ML ST第2/NT第5，未通过逐对差异检验、不称ST特异',
'PC':'次要前因：SEM两任务均ns，ML贡献中后段',
'WCE':'次要前因：SEM NT弱显著/ST边缘，ML贡献中段',
'PSC':'弱且偏负：SEM→ST=-.198*，ML两任务垫底、EN系数为负，Q42主观也排末位，适合Discussion'}
integ=[]
for f in Xcols:
    bn,sn,bs,ss=sem[f]
    integ.append(dict(代码=f,因素=cn[f],SEM到NT_β=bn,SEM到NT显著=sn,SEM到ST_β=bs,SEM到ST显著=ss,
        NT模型SHAP排名=ntrank[f],ST模型SHAP排名=strank[f],SHAP预测偏向=role(f),结合SEM结论=concl[f]))
integ=pd.DataFrame(integ)
# ---------- ⑤ Q42 三方对照 ----------
qrank={r['因素']:r['主观优先排名'] for _,r in q42.iterrows()}
qtot={r['因素']:r['进入前二总'] for _,r in q42.iterrows()}
qmap={'PARV':'PARV','PFE':'PFE','WCE':'WCE','FS':'FS','PSC':'PSC'}
rows=[]
for f in['PARV','PFE','WCE','FS','PSC','PC']:
    rows.append(dict(因素代码=f,因素=cn[f],Q42主观优先排名=qrank.get(qmap.get(f,''),'—（Q42无对应项）'),
        Q42进入前二人数=qtot.get(qmap.get(f,''),'—'),NT模型SHAP排名=ntrank[f],ST模型SHAP排名=strank[f]))
rows.append(dict(因素代码='NT一致',因素='分数和自己感觉一致(NT一致性,非6前因)',Q42主观优先排名=qrank.get('NT一致'),
    Q42进入前二人数=qtot.get('NT一致'),NT模型SHAP排名='—',ST模型SHAP排名='—'))
tri=pd.DataFrame(rows)
# ---------- 说明 ----------
note=pd.DataFrame({'口径与方法说明':[
 '样本N=274；6前因与NT/ST均由所属题项取平均得到构念分(1-5)，无反向题；ID不进模型。',
 '两个平行任务：PC,FS,PFE,WCE,PSC,PARV -> NT；同样6个 -> ST，输入/CV/算法/选参原则/指标完全一致。',
 '交叉验证：Repeated Nested 5-fold，外层5折×10重复=50个测试折评价泛化；内层5折同时为ElasticNet和XGBoost选超参；随机种子SEED=2026。',
 'R2_CV/RMSE/MAE均为外层未见样本(out-of-fold)口径；R2_CV是“对未见样本的预测力”，与SEM样本内R2(NT=.358,ST=.562)不是同一指标，通常更低、属正常。',
 'SHAP基于每折外层模型在其测试折上计算并汇总50折：平均排名、进Top2次数(/50)衡量排名稳定性；另用全量模型给出最终mean|SHAP|与带符号方向。',
 'Elastic Net为线性基准、XGBoost为非线性主模型；两者性能接近说明前因-信任关系以线性为主，并非失败。',
 'Q42不进入任何预测模型(避免概念重叠/信息泄漏)，仅在ML结束后作“显性主观优先级 vs SHAP预测贡献”的辅助对照。',
 'ML与SEM为同一批274人，只能称“预测层面的补充验证/convergent predictive evidence”，不能称独立验证；ML结果不回改SEM。',
 'SHAP方向措辞用“associated with higher predicted NT/ST”，不写因果(causes)。']})
# ---------- 写Excel ----------
xp=OUT+r'\机器学习结果汇总_ElasticNet_XGBoost_SHAP.xlsx'
with pd.ExcelWriter(xp,engine='openpyxl') as w:
    perf.to_excel(w,sheet_name='①模型性能CV',index=False)
    sNT.to_excel(w,sheet_name='②SHAP贡献稳定性_NT',index=False)
    sST.to_excel(w,sheet_name='③SHAP贡献稳定性_ST',index=False)
    integ.to_excel(w,sheet_name='④SHAP与SEM整合对照',index=False)
    tri.to_excel(w,sheet_name='⑤Q42三方对照',index=False)
    note.to_excel(w,sheet_name='⑥口径与方法说明',index=False)
print('[saved]',xp)
# ================= 图 =================
def bar_compare():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for k,t in enumerate(['NT','ST']):
        d={x['feat']:x['mean_abs_shap'] for x in R['targets'][t]['final_importance']}
        order=sorted(Xcols,key=lambda f:d[f]);lab=[f+' '+cn[f] for f in order];v=[d[f] for f in order]
        c=BLACK
        ax[k].barh(range(len(v)),v,color=c);ax[k].set_yticks(range(len(v)),lab,fontsize=9)
        ax[k].set_title('%s模型  XGBoost 平均|SHAP|贡献'%t,fontsize=11)
        for i,vv in enumerate(v):ax[k].text(vv,i,' %.3f'%vv,va='center',fontsize=8)
        ax[k].set_xlabel('mean(|SHAP|)')
    plt.tight_layout();plt.savefig(OUT+r'\图1_SHAP全局贡献_NT对比ST.png',dpi=160);plt.close()
def perf_fig():
    fig,ax=plt.subplots(1,3,figsize=(11,3.6))
    metrics=['R2_CV','RMSE','MAE'];titles=['R²(CV,越高越好)','RMSE(越低越好)','MAE(越低越好)']
    x=np.arange(2);w=.35
    for a,(mm,tt) in enumerate(zip(metrics,titles)):
        en=[perf[(perf.预测目标==t)&(perf.模型.str.contains('Elastic'))][mm].iloc[0] for t in['NT','ST']]
        xb=[perf[(perf.预测目标==t)&(perf.模型.str.contains('XGB'))][mm].iloc[0] for t in['NT','ST']]
        ax[a].bar(x-w/2,en,w,label='Elastic Net',color=GRAY_L,edgecolor='.45',linewidth=.6)
        ax[a].bar(x+w/2,xb,w,label='XGBoost',color=BLACK)
        ax[a].set_xticks(x,['NT','ST']);ax[a].set_title(tt,fontsize=10);ax[a].legend(fontsize=8)
        for i in x:
            ax[a].text(i-w/2,en[i],'%.3f'%en[i],ha='center',va='bottom',fontsize=7)
            ax[a].text(i+w/2,xb[i],'%.3f'%xb[i],ha='center',va='bottom',fontsize=7)
    plt.tight_layout();plt.savefig(OUT+r'\图2_两模型预测性能对比.png',dpi=160);plt.close()
def beeswarm(t):
    sv=pd.read_csv(OUT+r'\shap_full_%s.csv'%t).drop(columns=['target']).values
    fig=plt.figure(figsize=(8,4.5))
    shap.summary_plot(sv,X,feature_names=[f+' '+cn[f] for f in Xcols],cmap=gray_cmap,show=False,plot_size=(8,4.2))
    for coll in plt.gca().collections:
        coll.set_sizes([40]);coll.set_alpha(.92)
    plt.title('%s模型 SHAP Summary（深色=该前因取值高，浅色=低；横轴正=抬高预测%s）'%(t,t),fontsize=10)
    plt.tight_layout();plt.savefig(OUT+r'\图3_SHAP方向beeswarm_%s.png'%t,dpi=160,bbox_inches='tight');plt.close()
def stab_fig():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for k,t in enumerate(['NT','ST']):
        s=pd.DataFrame(R['targets'][t]['stability']).set_index('feat').loc[Xcols]
        y=np.arange(len(Xcols));c=BLACK
        ax[k].barh(y,s['top2_pct'],color=c);ax[k].set_yticks(y,[f for f in Xcols])
        ax[k].set_xlim(0,100);ax[k].set_xlabel('50个外层折中进入贡献Top2的比例 %')
        ax[k].set_title('%s模型 SHAP排名稳定性'%t);ax[k].invert_yaxis()
        for i,v in enumerate(s['top2_pct']):ax[k].text(v,i,' %d/50'%round(v/100*50),va='center',fontsize=8)
    plt.tight_layout();plt.savefig(OUT+r'\图4_SHAP排名稳定性_Top2次数.png',dpi=160);plt.close()
bar_compare();perf_fig();beeswarm('NT');beeswarm('ST');stab_fig()
print('[saved] 4组PNG图')
print('\n=== 性能 ===\n',perf.to_string(index=False))
print('\n=== SHAP×SEM整合 ===\n',integ.to_string(index=False))
print('\n=== Q42三方对照 ===\n',tri.to_string(index=False))
