# -*- coding: utf-8 -*-
"""额外生成一组【彩色·中文】SHAP图（参考标准SHAP配色：蓝Low→紫→品红High）。
只画图，不重写Excel；文件名加 _彩色，不覆盖灰度版/英文版。"""
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
import shap
from matplotlib.colors import LinearSegmentedColormap
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
cn={'PC':'感知控制','FS':'反馈稳定性','PFE':'感知反馈可解释性','WCE':'佩戴与校准体验','PSC':'隐私与监控担忧','PARV':'感知注意力调节价值'}
# 标准SHAP彩色：蓝(低)->紫->品红(高)；条形用SHAP经典蓝；性能图蓝/橙区分
SHAP_CMAP=LinearSegmentedColormap.from_list('shapcolor',['#1E9BFF','#8B5CF6','#FF1F6E'])
SHAP_BLUE='#1E88E5';C_EN='#4C78A8';C_XGB='#F58518'
R=json.load(open(OUT+r'\ml_nested_results_full.json',encoding='utf-8'))
data=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=data[Xcols].values.astype(float)
perf=[]
for t in['NT','ST']:
    for m,mn in[('EN','Elastic Net'),('XGB','XGBoost')]:
        d=R['targets'][t]['perf'][m];perf.append(dict(t=t,m=mn,**d))
perf=pd.DataFrame(perf)

def bar_compare():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for k,t in enumerate(['NT','ST']):
        d={x['feat']:x['mean_abs_shap'] for x in R['targets'][t]['final_importance']}
        order=sorted(Xcols,key=lambda f:d[f]);lab=[f+' '+cn[f] for f in order];v=[d[f] for f in order]
        ax[k].barh(range(len(v)),v,color=SHAP_BLUE);ax[k].set_yticks(range(len(v)),lab,fontsize=9)
        ax[k].set_title('%s模型  XGBoost 平均|SHAP|贡献'%t,fontsize=11)
        for i,vv in enumerate(v):ax[k].text(vv,i,' %.3f'%vv,va='center',fontsize=8)
        ax[k].set_xlabel('mean(|SHAP|)')
    plt.tight_layout();plt.savefig(OUT+r'\图1_SHAP全局贡献_NT对比ST_彩色.png',dpi=160);plt.close()

def perf_fig():
    fig,ax=plt.subplots(1,3,figsize=(11,3.6))
    mets=[('R2','R²(CV,越高越好)'),('RMSE','RMSE(越低越好)'),('MAE','MAE(越低越好)')]
    x=np.arange(2);w=.35
    for a,(mm,tt) in enumerate(mets):
        en=[perf[(perf.t==t)&(perf.m=='Elastic Net')][mm].iloc[0] for t in['NT','ST']]
        xb=[perf[(perf.t==t)&(perf.m=='XGBoost')][mm].iloc[0] for t in['NT','ST']]
        ax[a].bar(x-w/2,en,w,label='Elastic Net',color=C_EN,edgecolor='.3',linewidth=.6)
        ax[a].bar(x+w/2,xb,w,label='XGBoost',color=C_XGB)
        ax[a].set_xticks(x,['NT','ST']);ax[a].set_title(tt,fontsize=10);ax[a].legend(fontsize=8)
        for i in x:
            ax[a].text(i-w/2,en[i],'%.3f'%en[i],ha='center',va='bottom',fontsize=7)
            ax[a].text(i+w/2,xb[i],'%.3f'%xb[i],ha='center',va='bottom',fontsize=7)
    plt.tight_layout();plt.savefig(OUT+r'\图2_两模型预测性能对比_彩色.png',dpi=160);plt.close()

def beeswarm(t):
    sv=pd.read_csv(OUT+r'\shap_full_%s.csv'%t).drop(columns=['target']).values
    plt.figure(figsize=(8,4.5))
    shap.summary_plot(sv,X,feature_names=[f+' '+cn[f] for f in Xcols],cmap=SHAP_CMAP,show=False,plot_size=(8,4.2))
    for coll in plt.gca().collections:coll.set_sizes([40]);coll.set_alpha(.95)
    plt.title('%s模型 SHAP Summary（蓝=该前因取值低，品红=高；横轴正=抬高预测%s）'%(t,t),fontsize=10)
    plt.tight_layout();plt.savefig(OUT+r'\图3_SHAP方向beeswarm_%s_彩色.png'%t,dpi=160,bbox_inches='tight');plt.close()

def stab_fig():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for k,t in enumerate(['NT','ST']):
        s=pd.DataFrame(R['targets'][t]['stability']).set_index('feat').loc[Xcols]
        y=np.arange(len(Xcols))
        ax[k].barh(y,s['top2_pct'],color=SHAP_BLUE);ax[k].set_yticks(y,[f for f in Xcols])
        ax[k].set_xlim(0,100);ax[k].set_xlabel('50个外层折中进入贡献Top2的比例 %')
        ax[k].set_title('%s模型 SHAP排名稳定性'%t);ax[k].invert_yaxis()
        for i,v in enumerate(s['top2_pct']):ax[k].text(v,i,' %d/50'%round(v/100*50),va='center',fontsize=8)
    plt.tight_layout();plt.savefig(OUT+r'\图4_SHAP排名稳定性_Top2次数_彩色.png',dpi=160);plt.close()

bar_compare();perf_fig();beeswarm('NT');beeswarm('ST');stab_fig()
print('[saved] 5 张彩色中文图（_彩色）')
