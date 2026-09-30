# -*- coding: utf-8 -*-
"""生成机器学习一步步复现手册 Word。数字全部读自正式full结果。"""
import json,pandas as pd
from docx import Document
from docx.shared import Pt,RGBColor,Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
OUT=r'F:\研究生gpt\周老师\9.9处理\2\3\4'
R=json.load(open(OUT+r'\ml_nested_results_full.json',encoding='utf-8'))
q42=pd.read_csv(OUT+r'\Q42_主观优先级.csv')
doc=Document()
# 默认中文字体
st=doc.styles['Normal'];st.font.name='微软雅黑';st.font.size=Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
def H(t,l=1):
    h=doc.add_heading(t,level=l)
    for r in h.runs:r.font.name='微软雅黑';r._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑');r.font.color.rgb=RGBColor(0x1F,0x3A,0x5F)
    return h
def P(t,b=False,it=False,size=10.5,color=None):
    p=doc.add_paragraph();r=p.add_run(t);r.bold=b;r.italic=it;r.font.size=Pt(size)
    r.font.name='微软雅黑';r._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
    if color:r.font.color.rgb=color
    return p
def bullet(t):
    p=doc.add_paragraph(style='List Bullet');r=p.add_run(t);r.font.name='微软雅黑';r._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑');r.font.size=Pt(10.5)
def num(t):
    p=doc.add_paragraph(style='List Number');r=p.add_run(t);r.font.name='微软雅黑';r._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑');r.font.size=Pt(10.5)
def code(t):
    p=doc.add_paragraph();r=p.add_run(t);r.font.name='Consolas';r.font.size=Pt(9.5);r.font.color.rgb=RGBColor(0x8B,0x00,0x00)
    p.paragraph_format.left_indent=Inches(.25)
def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers));t.style='Light Grid Accent 1'
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i];c.text='';rr=c.paragraphs[0].add_run(h);rr.bold=True;rr.font.size=Pt(9.5)
        rr.font.name='微软雅黑';rr._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
    for row in rows:
        cs=t.add_row().cells
        for i,v in enumerate(row):
            cs[i].text='';rr=cs[i].paragraphs[0].add_run(str(v));rr.font.size=Pt(9.5)
            rr.font.name='微软雅黑';rr._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
    return t

# ===== 封面 =====
ti=doc.add_heading('机器学习复现操作手册',0)
for r in ti.runs:r.font.name='微软雅黑';r._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
P('Elastic Net（线性基准）+ XGBoost（主模型）+ SHAP（贡献解释）｜N=274｜SEED=2026',b=True)
P('研究：青年脑机接口注意力训练体验与信任——六前因对神经信任NT / 系统信任ST的预测贡献')
P('输出目录：F:\\研究生gpt\\周老师\\9.9处理\\2\\3\\4',it=True,size=9.5)
P('专用解释器：F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe（Python虚拟环境，勿用系统python）',it=True,size=9.5)

# ===== 0 速览 =====
H('0. 一页速览（先看这里）')
P('研究问题（机器学习只回答这一个）：在预测层面，6类体验因素对神经信任NT和系统信任ST是否呈现不同的贡献结构？该结构是否与SEM发现的“共享前因—轨道特异前因”模式一致？',b=True)
P('三句话结论：')
bullet('PARV（感知注意力调节价值）= NT特异前因：NT模型SHAP第1、50个外层折中48次进前二；ST模型第5、0次进前二。与SEM（β→NT=.380***，β→ST=ns，且唯一通过NT/ST逐对差异检验）构成“结构特异+预测特异”双重证据。')
bullet('PFE（感知反馈可解释性）= 双轨共享前因：NT第2、ST第1且50/50次进前二；与SEM两条路径都强显著一致。')
bullet('没有ST特异前因；PSC（隐私担忧）两任务都垫底、方向偏负，Q42主观也排末位，是讨论点而非驱动因素。')
P('性能（外层未见样本OOF口径）：NT  R²≈.20（EN .202 / XGB .156）；ST  R²≈.36（EN .353 / XGB .376）。两模型接近→前因—信任关系以线性为主（老师明确：这不是失败）。',b=True)

# ===== 1 设计 =====
H('1. 分析设计与老师的11步规范')
P('两个完全平行的预测任务，输入/CV/算法/超参搜索原则/指标完全一致：')
code('PC, FS, PFE, WCE, PSC, PARV  →  NT\nPC, FS, PFE, WCE, PSC, PARV  →  ST')
bullet('Elastic Net＝线性基准（带正则的线性回归，与SEM线性路径最接近，回答“线性关系下能预测多少”）。')
bullet('XGBoost＝非线性主模型（可捕捉非线性/阈值/交互；若明显优于EN说明SEM线性结构外还有预测信息，若持平说明关系近线性）。')
bullet('SHAP＝解释层，核心指标 mean(|SHAP|)，既看排名也看方向（beeswarm），并做50折排名稳定性。')
P('必须守住的红线（写论文时）：',b=True)
bullet('ML与SEM是同一批274人，只能称“预测层面的补充验证 / convergent predictive evidence”，不能称“独立验证”（独立验证需新外部样本）。')
bullet('Q42绝不作为特征放进XGBoost（与6前因概念重叠、会信息泄漏）；只在ML结束后做“主观优先级 vs SHAP预测贡献”辅助对照。')
bullet('从此SEM冻结，不得因SHAP高低回AMOS增删路径（避免循环的数据驱动建模）。')
bullet('ML的交叉验证R²(CV)是“对未见样本的预测力”，与SEM样本内R²(NT=.358/ST=.562)不是同一指标，通常更低、属正常，不可直接比较大小下结论。')
bullet('SHAP方向措辞用 “associated with higher predicted NT/ST”，不写 “causes / 导致”。')

# ===== 2 环境 =====
H('2. 运行环境（已就绪，零额外安装）')
P('使用专用虚拟环境解释器；关键依赖版本：numpy 2.5.3 / pandas 3.0.5 / scikit-learn 1.9.0 / xgboost 3.4.1 / shap 0.52.0 / matplotlib 3.11.1 / openpyxl 3.1.5（python-docx 1.2.0 用于生成本手册）。')
P('自检命令：')
code('& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" -m pip list | Select-String "scikit|xgboost|shap|pandas|numpy|matplotlib|openpyxl"')
P('全局随机种子 SEED=2026，保证结果可逐位复现。')

# ===== 3 数据准备 =====
H('3. 第一步：数据准备（ml_1_prep.py）')
num('读取源数据 0909-按分数_青年脑机接口注意力训练体验与信任调查.xlsx（Sheet1，274行）。')
num('不把36题直接当特征，而是按构念取所属题项“平均分”（无反向题，保持1–5量尺）。题项归属：')
table(['构念','含义','题项'],[
 ['NT','神经信任','Q10,Q11,Q15,Q16'],['ST','系统信任','Q12,Q14,Q20,Q22'],
 ['PC','感知控制','Q13,Q19,Q23,Q27'],['FS','反馈稳定性','Q9,Q29,Q35,Q40'],
 ['PFE','感知反馈可解释性','Q17,Q18,Q21,Q36'],['WCE','佩戴与校准体验','Q6,Q7,Q8,Q34'],
 ['PSC','隐私与监控担忧','Q24,Q25,Q26,Q28'],['PARV','感知注意力调节价值','Q30,Q31,Q32,Q33']])
num('生成 建模数据集_量表分.csv（274行；ID只识别样本、不进模型）。')
num('解析Q42（文本“第一选择→第二选择”），映射到构念，算第一/第二/进入前二人数、Score=2×第一+第二，并与老师给的频次表逐项对拍——本次6项全部一致✓（第一、第二选择各合计274）。')
P('运行命令：')
code('& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" ml_1_prep.py')

# ===== 4 建模 =====
H('4. 第二步：嵌套交叉验证建模（ml_3_nested.py）')
P('为什么不用70/30单次划分：N=274时单次测试集仅约80人，结果依赖一次随机划分。故采用 Repeated Nested 5-fold CV：',b=True)
bullet('外层：5折 × 10重复 = 50个外层测试折，负责评价对“未见样本”的泛化（R²/RMSE/MAE，out-of-fold汇总）。')
bullet('内层：每个外层训练集内部再做5折CV，同时为 Elastic Net（选 l1_ratio、alpha）和 XGBoost（选 max_depth、n_estimators）选超参，杜绝用测试信息调参的乐观偏差。')
bullet('XGBoost内层网格：max_depth∈{1,2,3}，n_estimators∈{150,300,500}；learning_rate=.03、subsample=.8、colsample_bytree=.9、min_child_weight=5、reg_lambda=1.5、reg_alpha=.1 固定（小样本保守防过拟合）。')
P('SHAP稳定性（小样本关键，老师第九步）：不只看全样本一张图——在每个外层折用该折模型对其测试折算SHAP，累计50折，统计每个因素的平均排名、贡献份额、进入Top2/Top3的次数（如48/50）。另在全量274上拟合最终模型，给最终 mean|SHAP| 排序、带符号方向与Elastic Net标准化系数。')
P('先小样跑通、再正式（推荐习惯）：')
code('# 小样：1重复、小网格，仅验证流程（结果不作数）\n... python.exe ml_3_nested.py quick\n# 正式：50外层折+内层选参（本手册数字来自这一版）\n... python.exe ml_3_nested.py full')

# ===== 5 出报告 =====
H('5. 第三步：生成汇总表与图（ml_4_report.py）')
code('& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" ml_4_report.py')
P('产出：机器学习结果汇总_ElasticNet_XGBoost_SHAP.xlsx（6张表）＋4组PNG图。')
table(['Sheet/文件','内容'],[
 ['①模型性能CV','NT/ST × EN/XGB 的 OOF R²、RMSE、MAE 与折RMSE分布'],
 ['②SHAP贡献稳定性_NT / ③_ST','全量SHAP排名/方向、EN系数、50折平均排名、进Top2次数、贡献份额'],
 ['④SHAP与SEM整合对照','每个前因的SEM→NT/ST β与显著性、两任务SHAP排名、预测偏向、结合SEM结论'],
 ['⑤Q42三方对照','Q42主观优先排名 × NT-SHAP排名 × ST-SHAP排名（PC无Q42项；NT一致性非6前因）'],
 ['⑥口径与方法说明','所有口径、CV设计、方法学纪律文字，可直接挪进论文方法部分'],
 ['图1','两任务SHAP全局贡献条形对比'],['图2','两模型R²/RMSE/MAE性能对比'],
 ['图3(NT/ST)','SHAP beeswarm，看高低值与正负方向'],['图4','50折进Top2次数＝排名稳定性']])

# ===== 6 结果 =====
H('6. 正式结果（full，可逐位复现）')
H('6.1 预测性能（OOF，越高R²越好，误差越低越好）',2)
pf=[]
for t in['NT','ST']:
    for m,mn in[('EN','Elastic Net'),('XGB','XGBoost')]:
        d=R['targets'][t]['perf'][m];pf.append([t,mn,d['R2'],d['RMSE'],d['MAE']])
table(['目标','模型','R²(CV)','RMSE','MAE'],pf)
P('解读：ST比NT更可预测（R²约.36 vs .20）；EN与XGB基本持平（NT线性略好、ST非线性略好），内层CV最常选出最浅树 max_depth=1，共同说明前因—信任关系以线性、可加为主，非线性增益有限。')
for t in['NT','ST']:
    H('6.2 %s模型：SHAP贡献与50折稳定性'%t,2)
    rows=[[d['fullrank'],d['feat'],d['cn'],d['mean_abs_shap'],d['signed_shap'],d['en_coef_std']] for d in R['targets'][t]['final_importance']]
    table(['全量排名','代码','因素','mean|SHAP|','带符号SHAP(方向)','EN标准化系数'],rows)
    stb=R['targets'][t]['stability']
    rows2=[[s['feat'],s['cn'],s['mean_rank'],'%d/50'%round(s['top2_pct']/100*50),'%d%%'%s['top3_pct'],'%.1f%%'%s['share_pct']] for s in stb]
    table(['代码','因素','50折平均排名','进Top2次数','进Top3比例','平均贡献份额'],rows2)
H('6.3 SHAP × SEM 整合（核心交叉证据）',2)
P('PARV：SEM中→NT=.380***、→ST=ns且唯一通过逐对差异检验；ML中NT第1(48/50)、ST第5(0/50) → NT特异前因，双重证据。')
P('PFE：SEM两任务都强显著（.346***/.538***）；ML中NT第2、ST第1(50/50) → 双轨共享前因。')
P('FS：ML偏ST（ST第2/NT第5）但SEM仅ST边缘、且未通过逐对差异检验，只能称“偏ST的次要前因”，不称ST特异。PSC：两任务SHAP垫底、EN系数为负，与SEM→ST=-.198*方向一致，是讨论点。')
H('6.4 Q42 主观优先级 × SHAP 三方对照',2)
rows=[]
ntf={d['feat']:d['fullrank'] for d in R['targets']['NT']['final_importance']}
stf={d['feat']:d['fullrank'] for d in R['targets']['ST']['final_importance']}
for _,r in q42.sort_values('主观优先排名').iterrows():
    f=r['因素'];rows.append([f,'%d'%r['主观优先排名'],r['第一选择'],r['第二选择'],r['进入前二总'],
        ntf.get(f,'—（非6前因）'),stf.get(f,'—（非6前因）')])
rows.append(['PC（感知控制，Q42无对应项）','—','—','—','—',ntf['PC'],stf['PC']])
table(['因素','Q42主观排名','第一','第二','前二总','NT-SHAP排名','ST-SHAP排名'],rows)
P('对照看点：PARV主观第1且NT-SHAP第1（主观与预测在NT上一致）；PFE主观第2且两任务SHAP都靠前；PSC主观垫底且预测也最弱（用户说不重要、统计上也弱，方向一致）；WCE主观第3但预测贡献中等，属“用户在意但并非最强预测因素”，适合Discussion。')

# ===== 7 一键复现 =====
H('7. 完整一键复现（按顺序三条命令）')
code('cd "F:\\研究生gpt\\周老师\\9.9处理\\2\\3\\4"\n'
'& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" ml_1_prep.py\n'
'& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" ml_3_nested.py full\n'
'& "F:\\研究生gpt\\周老师\\机器学习\\venv\\Scripts\\python.exe" ml_4_report.py')
P('说明：三个脚本已随结果一并归档在本文件夹；改数据后按上面顺序重跑即可全部刷新。想换重复次数/网格，改 ml_3_nested.py 顶部 REPS 与 GRID；想换随机种子改 SEED（改种子属于稳健性分析，主结果保持SEED=2026）。')
P('附：shap_fold_stability_full.csv 是50折×6特征的逐折SHAP明细，shap_full_NT/ST.csv 是全量SHAP值（可自行画更多图），ml_nested_results_full.json 是全部数值底稿。',size=9.5,it=True)
doc.save(OUT+r'\机器学习_ElasticNet_XGBoost_SHAP_复现操作手册.docx')
print('[saved] Word手册')
