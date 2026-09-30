# -*- coding: utf-8 -*-
"""OOF SHAP 补图：图1 OOF Mean|SHAP|条形 + 图3 OOF beeswarm(NT/ST)。
beeswarm用全部外层测试折SHAP(274x10=2740条,只在未见测试集上算)。四套语言x配色各3张。"""
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
cn={'PC':'感知控制','FS':'反馈稳定性','PFE':'感知反馈可解释性','WCE':'佩戴与校准体验','PSC':'隐私与监控担忧','PARV':'感知注意力调节价值'}
en={'PC':'Perceived Control (PC)','FS':'Feedback Stability (FS)','PFE':'Perceived Feedback Explainability (PFE)',
'WCE':'Wear & Calibration Experience (WCE)','PSC':'Privacy & Surveillance Concern (PSC)','PARV':'Perceived Attention Regulation Value (PARV)'}
gray=LinearSegmentedColormap.from_list('g',['#ececec','#a0a0a0','#000000'])
color=LinearSegmentedColormap.from_list('c',['#1E9BFF','#8B5CF6','#FF1F6E'])
SHAP_BLUE='#1E88E5';BLACK='#262626'
oof={t:pd.read_csv(OUT+r'\shap_oof_%s.csv'%t) for t in['NT','ST']}
styles=[
 ('cn','gray',cn,'Microsoft YaHei','sans-serif','',''),
 ('cn','color',cn,'Microsoft YaHei','sans-serif','_彩色',''),
 ('en','gray',en,'Times New Roman','serif','_EN','EN'),
 ('en','color',en,'Times New Roman','serif','_EN_彩色','EN'),
]
def run(lang,pal,labdict,font,family,suf,entag):
    plt.rcParams['font.family']=family
    if family=='serif':plt.rcParams['font.serif']=[font,'DejaVu Serif'];plt.rcParams['mathtext.fontset']='stix'
    else:plt.rcParams['font.sans-serif']=[font]
    plt.rcParams['axes.unicode_minus']=False
    cmap=gray if pal=='gray' else color;barc=BLACK if pal=='gray' else SHAP_BLUE
    # ---- 图1 OOF Mean|SHAP| ----
    fig,ax=plt.subplots(1,2,figsize=(11.5 if entag else 11,4.4))
    for k,t in enumerate(['NT','ST']):
        sv=oof[t][[f+'_shap' for f in Xcols]].values;m=np.abs(sv).mean(0)
        order=sorted(range(6),key=lambda i:m[i]);l=[labdict[Xcols[i]] for i in order];v=[m[i] for i in order]
        ax[k].barh(range(6),v,color=barc);ax[k].set_yticks(range(6),l,fontsize=(10 if entag else 9))
        ax[k].set_title(('%s Model — Out-of-Fold Mean |SHAP|'%t) if entag else ('%s模型  OOF平均|SHAP|贡献（外层测试折汇总）'%t),
                        fontsize=(12 if entag else 11),fontweight='bold' if entag else 'normal')
        ax[k].set_xlabel('mean(|SHAP|), out-of-fold',fontsize=11 if entag else 10)
        for i,vv in enumerate(v):ax[k].text(vv,i,' %.4f'%vv,va='center',fontsize=(9 if entag else 8))
    plt.tight_layout();plt.savefig(OUT+r'\图1_OOF_SHAP全局贡献_NT对比ST%s.png'%suf,dpi=(200 if entag else 160));plt.close()
    # ---- 图3 OOF beeswarm ----
    for t in['NT','ST']:
        d=oof[t];sv=d[[f+'_shap' for f in Xcols]].values;xx=d[[f+'_x' for f in Xcols]].values
        plt.figure(figsize=(8.5 if entag else 8,4.6 if entag else 4.5))
        shap.summary_plot(sv,xx,feature_names=[labdict[f] for f in Xcols],cmap=cmap,show=False,
                          plot_size=(8.5 if entag else 8,4.4 if entag else 4.2))
        for coll in plt.gca().collections:coll.set_sizes([16]);coll.set_alpha(.82)
        if entag:
            plt.title('%s Model Out-of-Fold SHAP (held-out folds only; %s = high feature value, right = raises predicted %s)'%(
                t,'red' if pal=='color' else 'black',t),fontsize=10)
            plt.xlabel('SHAP value (out-of-fold impact)',fontsize=12)
        else:
            plt.title('%s模型 OOF SHAP（仅对每折未见过的测试集计算；%s=取值高，横轴正=抬高预测%s）'%(
                t,'深色' if pal=='gray' else '品红',t),fontsize=10)
            plt.xlabel('SHAP value（外层测试折）',fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT+r'\图3_OOF_SHAP方向beeswarm_%s%s.png'%(t,suf),dpi=(200 if entag else 160),bbox_inches='tight');plt.close()
for s in styles:run(*s)
print('[saved] OOF补图 4套 x 3张 = 12张')
