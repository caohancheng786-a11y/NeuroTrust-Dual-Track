# -*- coding: utf-8 -*-
"""数据准备：36题 -> 构念量表分；解析Q42并与老师频次对拍。不建模。"""
import pandas as pd,numpy as np,json
SRC=r'F:\研究生gpt\周老师\9.9处理\2\0909-按分数_青年脑机接口注意力训练体验与信任调查.xlsx'
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
groups={'NT':['Q10','Q11','Q15','Q16'],'ST':['Q12','Q14','Q20','Q22'],
'PC':['Q13','Q19','Q23','Q27'],'FS':['Q9','Q29','Q35','Q40'],'PFE':['Q17','Q18','Q21','Q36'],
'WCE':['Q6','Q7','Q8','Q34'],'PSC':['Q24','Q25','Q26','Q28'],'PARV':['Q30','Q31','Q32','Q33'],
'RI':['Q37','Q38','Q39','Q41']}
df=pd.read_excel(SRC)
for g,qs in groups.items():
    df[g]=df[qs].apply(pd.to_numeric,errors='coerce').mean(axis=1)
Xcols=['PC','FS','PFE','WCE','PSC','PARV']
model=df[['序号']+Xcols+['NT','ST','RI']].copy()
model.to_csv(OUT+r'\建模数据集_量表分.csv',index=False,encoding='utf-8-sig')
print('建模数据集:',model.shape,'（274人；6前因+NT/ST/RI标签）')
print('\n各构念量表分 均值/标准差/最小/最大:')
print(model[Xcols+['NT','ST']].agg(['mean','std','min','max']).T.round(3))

# ---- Q42 解析 ----
qmap={'训练真的能帮助自己觉察或调整注意力':'PARV','报告能解释清楚为什么得这个分':'PFE',
'佩戴舒适，不影响训练':'WCE','校准顺利，信号稳定':'FS','分数和自己的感觉一致':'NT一致',
'数据隐私有保障，不会被别人随便看到':'PSC'}
def split_q(v):
    a=str(v).split('→');return (a[0].strip(),a[1].strip()) if len(a)==2 else (a[0].strip(),None)
r1=df['Q42'].map(lambda v:qmap[split_q(v)[0]]);r2=df['Q42'].map(lambda v:qmap[split_q(v)[1]])
fac=['PARV','PFE','WCE','FS','NT一致','PSC']
rows=[]
for fac_ in fac:
    n1=int((r1==fac_).sum());n2=int((r2==fac_).sum())
    rows.append(dict(因素=fac_,第一选择=n1,第二选择=n2,进入前二总=n1+n2,Score=2*n1+n2))
q42=pd.DataFrame(rows).sort_values('进入前二总',ascending=False).reset_index(drop=True)
q42['主观优先排名']=range(1,len(q42)+1)
print('\n--- Q42 解析结果（按进入前二总数排序）---')
print(q42.to_string(index=False))
print('第一选择合计=%d（应=274），第二选择合计=%d（应=274）'%(q42.第一选择.sum(),q42.第二选择.sum()))
# 老师给的目标值
teacher={'PARV':(57,65,122),'PFE':(43,72,115),'WCE':(90,17,107),'FS':(33,52,85),'NT一致':(39,38,77),'PSC':(12,30,42)}
ok=True
print('\n--- 与老师频次对拍 ---')
for fac_,(t1,t2,tt) in teacher.items():
    row=q42[q42.因素==fac_].iloc[0];a=(row.第一选择,row.第二选择,row.进入前二总)
    good=a==(t1,t2,tt);ok=ok and good
    print(f'{fac_:6s} 我解析{a} 老师{(t1,t2,tt)} {"✓" if good else "✗ 不一致!"}')
print('\nQ42解析与老师完全一致:',ok)
q42.to_csv(OUT+r'\Q42_主观优先级.csv',index=False,encoding='utf-8-sig')
# 6前因/NT/ST相关矩阵（供与SEM方向对照）
print('\n--- 预测变量与NT/ST的Pearson相关（线性方向，供对照SEM β）---')
print(model[Xcols+['NT','ST']].corr().loc[Xcols,['NT','ST']].round(3))
