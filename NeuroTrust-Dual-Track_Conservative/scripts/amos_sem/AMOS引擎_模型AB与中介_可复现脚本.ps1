# =====================================================================
# AMOS 第1层(模型A vs 模型B) + 第3层(完全中介 vs 部分中介)
# 调用 IBM SPSS Amos 31 引擎，一次会话拟合三个嵌套模型，与 GUI 同内核。
# 数据：同目录 跑.sav（N=274）。运行：powershell -ExecutionPolicy Bypass -File 本脚本.ps1
# =====================================================================
$ErrorActionPreference="Stop"
$amosDir="F:\Program Files\IBM\SPSS\Amos\31"
$here=Split-Path -Parent $MyInvocation.MyCommand.Path
$work=Join-Path $env:TEMP "amoswork2"; New-Item -ItemType Directory -Force $work|Out-Null
Copy-Item (Join-Path $here "跑.sav") (Join-Path $work "data.sav") -Force
$sav=Join-Path $work "data.sav"
$scratch="C:\Users\ASUS\AppData\Local\Temp\AmosTemp\AmosScratch.AmosOutput"
if(Test-Path $scratch){Remove-Item $scratch -Force}
Add-Type -Path (Join-Path $amosDir "Amos.EngineLib.dll")
$sem=New-Object AmosEngineLib.AmosEngine
$L=New-Object System.Collections.Generic.List[string]
$m=@(
 "Q10 = (1) NT + (1) e4","Q11 = NT + (1) e3","Q15 = NT + (1) e2","Q16 = NT + (1) e1",
 "Q12 = ST + (1) e8","Q14 = ST + (1) e7","Q20 = ST + (1) e6","Q22 = (1) ST + (1) e5",
 "Q13 = PC + (1) e12","Q19 = PC + (1) e11","Q23 = PC + (1) e10","Q27 = (1) PC + (1) e9",
 "Q29 = FS + (1) e15","Q35 = FS + (1) e14","Q40 = (1) FS + (1) e13","Q9  = FS + (1) e16",
 "Q21 = PFE + (1) e18","Q36 = (1) PFE + (1) e17","Q18 = PFE + (1) e19","Q17 = PFE + (1) e20",
 "Q6  = WCE + (1) e24","Q7  = WCE + (1) e23","Q8  = WCE + (1) e22","Q34 = (1) WCE + (1) e21",
 "Q24 = PSC + (1) e28","Q25 = PSC + (1) e27","Q26 = PSC + (1) e26","Q28 = (1) PSC + (1) e25",
 "Q30 = PARV + (1) e32","Q31 = PARV + (1) e31","Q32 = PARV + (1) e30","Q33 = (1) PARV + (1) e29",
 "Q37 = RI + (1) e36","Q38 = RI + (1) e35","Q39 = RI + (1) e34","Q41 = (1) RI + (1) e33")
foreach($x in $m){$L.Add($x)}
$L.Add("NT = (pc_nt) PC + (fs_nt) FS + (pfe_nt) PFE + (wce_nt) WCE + (psc_nt) PSC + (parv_nt) PARV + (1) zNT")
$L.Add("ST = (pc_st) PC + (fs_st) FS + (pfe_st) PFE + (wce_st) WCE + (psc_st) PSC + (parv_st) PARV + (1) zST")
$L.Add("RI = (pc_ri) PC + (fs_ri) FS + (pfe_ri) PFE + (wce_ri) WCE + (psc_ri) PSC + (parv_ri) PARV + (nt_ri) NT + (st_ri) ST + (1) zRI")
foreach($v in @("PC","FS","PFE","WCE","PSC","PARV")){$L.Add("$v (v$v)")}
$L.Add("zNT (vzNT)");$L.Add("zST (vzST)");$L.Add("zRI (vzRI)");$L.Add("zNT <> zST")
for($i=1;$i -le 36;$i++){$L.Add("e$i (ve$i)")}
$exo=@("PC","FS","PFE","WCE","PSC","PARV")
for($a=0;$a -lt $exo.Count;$a++){for($b=$a+1;$b -lt $exo.Count;$b++){$L.Add("$($exo[$a]) <> $($exo[$b])")}}
$sem.TextOutput($true);$sem.Standardized($true);$sem.BeginGroup($sav)
foreach($x in $L){$sem.AStructure($x)}
[string[]]$zA=@("pc_ri = 0","fs_ri = 0","pfe_ri = 0","wce_ri = 0","psc_ri = 0","parv_ri = 0","pfe_nt = 0","wce_nt = 0","psc_nt = 0","pc_st = 0","fs_st = 0")
[string[]]$zB=@("pc_ri = 0","fs_ri = 0","pfe_ri = 0","wce_ri = 0","psc_ri = 0","parv_ri = 0")
$sem.Model("ModelA",$zA);$sem.Model("ModelB_FullMed",$zB);$sem.Model("Partial",([string[]]@()))
foreach($mn in @("ModelA","ModelB_FullMed","Partial")){$sem.FitModel($mn)|Out-Null;Write-Host ("{0}: chi2={1:F4} df={2} NPAR={3}" -f $mn,$sem.Cmin(),$sem.Df(),$sem.NumberOfParameters())}
$sem.Dispose()
for($k=0;$k -lt 30;$k++){Start-Sleep -Milliseconds 500;if(Test-Path $scratch){$c=Get-Content $scratch -Raw;if($c -match "1184.16" -and $c -match "1256.01" -and $c -match "1280.32"){break}}}
Copy-Item $scratch (Join-Path $here "AMOS_模型AB与中介_原始输出报告.html") -Force
Write-Host ("完成，报告已输出到 "+$here)
