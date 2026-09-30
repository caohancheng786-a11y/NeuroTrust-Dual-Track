# -*- coding: utf-8 -*-
"""英文·彩色版 SHAP图（Times New Roman + 标准SHAP蓝->紫->品红）。文件名 _EN_彩色，不覆盖既有图。"""
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family']='serif';plt.rcParams['font.serif']=['Times New Roman','DejaVu Serif']
plt.rcParams['mathtext.fontset']='stix';plt.rcParams['axes.unicode_minus']=False
import shap
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
en={'PC':'Perceived Control (PC)','FS':'Feedback Stability (FS)','PFE':'Perceived Feedback Explainability (PFE)',
'WCE':'Wear & Calibration Experience (WCE)','PSC':'Privacy & Surveillance Concern (PSC)','PARV':'Perceived Attention Regulation Value (PARV)'}
SHAP_CMAP=LinearSegmentedColormap.from_list('shapcolor',['#1E9BFF','#8B5CF6','#FF1F6E'])
SHAP_BLUE='#1E88E5';C_EN='#4C78A8';C_XGB='#F58518'
R=json.load(open(OUT+r'\ml_nested_results_full.json',encoding='utf-8'))
data=pd.read_csv(OUT+r'\建模数据集_量表分.csv');X=data[Xcols].values.astype(float)
lab=[en[f] for f in Xcols]
SUF='_EN_彩色.png'

def bar_compare():
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
    for k,t in enumerate(['NT','ST']):
        d={x['feat']:x['mean_abs_shap'] for x in R['targets'][t]['final_importance']}
        order=sorted(Xcols,key=lambda f:d[f]);l=[en[f] for f in order];v=[d[f] for f in order]
        ax[k].barh(range(len(v)),v,color=SHAP_BLUE);ax[k].set_yticks(range(len(v)),l,fontsize=10)
        ax[k].set_title('%s Model — Mean |SHAP| Contribution'%t,fontsize=12,fontweight='bold')
        for i,vv in enumerate(v):ax[k].text(vv,i,' %.3f'%vv,va='center',fontsize=9)
        ax[k].set_xlabel('mean(|SHAP|)',fontsize=11)
    plt.tight_layout();plt.savefig(OUT+r'\图1_SHAP全局贡献_NT对比ST'+SUF,dpi=200);plt.close()

def perf_fig():
    perf={(t,m):R['targets'][t]['perf'][m] for t in['NT','ST'] for m in('EN','XGB')}
    fig,ax=plt.subplots(1,3,figsize=(11.5,3.8));x=np.arange(2);w=.35
    mets=[('R2','R$^2$ (CV, higher better)'),('RMSE','RMSE (lower better)'),('MAE','MAE (lower better)')]
    for a,(mm,tt) in enumerate(mets):
        en_=[perf[(t,'EN')][mm] for t in['NT','ST']];xb=[perf[(t,'XGB')][mm] for t in['NT','ST']]
        ax[a].bar(x-w/2,en_,w,label='Elastic Net',color=C_EN,edgecolor='.3',linewidth=.6)
        ax[a].bar(x+w/2,xb,w,label='XGBoost',color=C_XGB)
        ax[a].set_xticks(x,['NT','ST'],fontsize=11);ax[a].set_title(tt,fontsize=11);ax[a].legend(fontsize=9)
        for i in x:
            ax[a].text(i-w/2,en_[i],'%.3f'%en_[i],ha='center',va='bottom',fontsize=8)
            ax[a].text(i+w/2,xb[i],'%.3f'%xb[i],ha='center',va='bottom',fontsize=8)
    plt.tight_layout();plt.savefig(OUT+r'\图2_两模型预测性能对比'+SUF,dpi=200);plt.close()

def beeswarm(t):
    sv=pd.read_csv(OUT+r'\shap_full_%s.csv'%t).drop(columns=['target']).values
    plt.figure(figsize=(8.5,4.6))
    shap.summary_plot(sv,X,feature_names=lab,cmap=SHAP_CMAP,show=False,plot_size=(8.5,4.4))
    for coll in plt.gca().collections:coll.set_sizes([40]);coll.set_alpha(.95)
    plt.title('%s Model SHAP Summary (red = high feature value, blue = low; right = raises predicted %s)'%(t,t),fontsize=10.5)
    plt.xlabel('SHAP value (impact on model output)',fontsize=12)
    plt.tight_layout();plt.savefig(OUT+r'\图3_SHAP方向beeswarm_%s'%t+SUF,dpi=200,bbox_inches='tight');plt.close()

def stab_fig():
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
    for k,t in enumerate(['NT','ST']):
        s=pd.DataFrame(R['targets'][t]['stability']).set_index('feat').loc[Xcols]
        y=np.arange(len(Xcols))
        ax[k].barh(y,s['top2_pct'],color=SHAP_BLUE);ax[k].set_yticks(y,Xcols,fontsize=11)
        ax[k].set_xlim(0,100);ax[k].set_xlabel('Share of 50 outer folds ranked in Top 2 (%)',fontsize=10.5)
        ax[k].set_title('%s Model — SHAP Rank Stability'%t,fontsize=12,fontweight='bold');ax[k].invert_yaxis()
        for i,v in enumerate(s['top2_pct']):ax[k].text(v,i,' %d/50'%round(v/100*50),va='center',fontsize=9)
    plt.tight_layout();plt.savefig(OUT+r'\图4_SHAP排名稳定性_Top2次数'+SUF,dpi=200);plt.close()

bar_compare();perf_fig();beeswarm('NT');beeswarm('ST');stab_fig()
print('[saved] 5 English COLOR figures (_EN_彩色)')
