# =====================================================================
# AMOS 自由模型 vs 等值约束模型  χ²差异检验（调用 IBM SPSS Amos 31 引擎）
# 与在 AMOS Graphics 里手动建模、点 "Calculate Estimates" 是同一计算内核，
# 输出的数值即 AMOS 官方结果。
# 数据：本脚本同目录下的 跑.sav（N=274）
# 运行：右键“使用 PowerShell 运行”，或 powershell -ExecutionPolicy Bypass -File 本脚本.ps1
# =====================================================================
$ErrorActionPreference = "Stop"
$amosDir  = "F:\Program Files\IBM\SPSS\Amos\31"
$here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$savSrc   = Join-Path $here "跑.sav"
$work     = Join-Path $env:TEMP "amoswork"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$sav = Join-Path $work "data.sav"
Copy-Item $savSrc $sav -Force

Add-Type -Path (Join-Path $amosDir "Amos.EngineLib.dll")
$sem = New-Object AmosEngineLib.AmosEngine
$L = New-Object System.Collections.Generic.List[string]

# ---- 测量模型（9因子36题，每因子固定1个载荷为1）----
$measure = @(
 "Q10 = (1) NT + (1) e4","Q11 = NT + (1) e3","Q15 = NT + (1) e2","Q16 = NT + (1) e1",
 "Q12 = ST + (1) e8","Q14 = ST + (1) e7","Q20 = ST + (1) e6","Q22 = (1) ST + (1) e5",
 "Q13 = PC + (1) e12","Q19 = PC + (1) e11","Q23 = PC + (1) e10","Q27 = (1) PC + (1) e9",
 "Q29 = FS + (1) e15","Q35 = FS + (1) e14","Q40 = (1) FS + (1) e13","Q9  = FS + (1) e16",
 "Q21 = PFE + (1) e18","Q36 = (1) PFE + (1) e17","Q18 = PFE + (1) e19","Q17 = PFE + (1) e20",
 "Q6  = WCE + (1) e24","Q7  = WCE + (1) e23","Q8  = WCE + (1) e22","Q34 = (1) WCE + (1) e21",
 "Q24 = PSC + (1) e28","Q25 = PSC + (1) e27","Q26 = PSC + (1) e26","Q28 = (1) PSC + (1) e25",
 "Q30 = PARV + (1) e32","Q31 = PARV + (1) e31","Q32 = PARV + (1) e30","Q33 = (1) PARV + (1) e29",
 "Q37 = RI + (1) e36","Q38 = RI + (1) e35","Q39 = RI + (1) e34","Q41 = (1) RI + (1) e33")
foreach($x in $measure){ $L.Add($x) }

# ---- 结构模型：6前因→NT/ST（命名参数），NT/ST→RI，内生潜变量带扰动项 ----
$L.Add("NT = (pc_nt) PC + (fs_nt) FS + (pfe_nt) PFE + (wce_nt) WCE + (psc_nt) PSC + (parv_nt) PARV + (1) zNT")
$L.Add("ST = (pc_st) PC + (fs_st) FS + (pfe_st) PFE + (wce_st) WCE + (psc_st) PSC + (parv_st) PARV + (1) zST")
$L.Add("RI = NT + ST + (1) zRI")
foreach($v in @("PC","FS","PFE","WCE","PSC","PARV")){ $L.Add("$v (v$v)") }
$L.Add("zNT (vzNT)"); $L.Add("zST (vzST)"); $L.Add("zRI (vzRI)"); $L.Add("zNT <> zST")
for($i=1;$i -le 36;$i++){ $L.Add("e$i (ve$i)") }
$exo=@("PC","FS","PFE","WCE","PSC","PARV")
for($a=0;$a -lt $exo.Count;$a++){ for($b=$a+1;$b -lt $exo.Count;$b++){ $L.Add("$($exo[$a]) <> $($exo[$b])") } }

$sem.TextOutput($true); $sem.Standardized($true); $sem.BeginGroup($sav)
foreach($x in $L){ $sem.AStructure($x) }
$sem.Model("Free",([string[]]@()))
[string[]]$cons=@("pc_nt = pc_st","fs_nt = fs_st","pfe_nt = pfe_st","wce_nt = wce_st","psc_nt = psc_st","parv_nt = parv_st")
$sem.Model("Constrained",$cons)

$sem.FitModel("Free") | Out-Null
$cF=$sem.Cmin(); $dF=$sem.Df()
$sem.FitModel("Constrained") | Out-Null
$cC=$sem.Cmin(); $dC=$sem.Df()
$report=$sem.TextOutputFileName()
$sem.Dispose()

$dChi=$cC-$cF; $dDf=$dC-$dF
Copy-Item $report (Join-Path $here "AMOS_自由vs约束_原始输出报告.html") -Force
$txt = "AMOS Chi-square Difference Test (N=274)`r`n"
$txt += "Free       : chi2=$($cF.ToString('F4')) df=$dF`r`n"
$txt += "Constrained: chi2=$($cC.ToString('F4')) df=$dC`r`n"
$txt += "DELTA      : dChi2=$($dChi.ToString('F4')) dDf=$dDf  (p<.001 => reject equality, dual-track supported)`r`n"
$txt | Out-File (Join-Path $here "AMOS_卡方差异检验_数值摘要.txt") -Encoding UTF8
Write-Host $txt
Write-Host ("完成。结果已输出到: " + $here)
