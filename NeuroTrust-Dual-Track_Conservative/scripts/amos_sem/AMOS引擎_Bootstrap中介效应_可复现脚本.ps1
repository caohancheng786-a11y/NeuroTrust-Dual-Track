# =====================================================================
# Reproducible AMOS Bootstrap mediation (Step 2 of teacher's SEM plan)
# 7 specific indirect effects, Bootstrap=5000, seed=2026, BC+PC 95% CI
# Model A (main) AND Partial-mediation (robustness) in one run.
# Outputs land next to this script when $OutDir is set below.
# Prereq: IBM SPSS Amos 31 installed at $AmosDir; data = pao.sav
# =====================================================================
$ErrorActionPreference = "Stop"
$AmosDir = "F:\Program Files\IBM\SPSS\Amos\31"
$DataSrc = "F:\研究生gpt\周老师\9.9处理\2\跑.sav"            # source SPSS data (N=274)
$OutDir  = "F:\研究生gpt\周老师\9.9处理\2"                   # deliverables folder
$Work    = "C:\Users\ASUS\AppData\Local\Temp\amoswork2"
$Scratch = "C:\Users\ASUS\AppData\Local\Temp\AmosTemp\AmosScratch.AmosOutput"
$NB = 5000; $SEED = 2026
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Copy-Item $DataSrc (Join-Path $Work "data.sav") -Force
$sav = Join-Path $Work "data.sav"
Add-Type -Path (Join-Path $AmosDir "Amos.EngineLib.dll")
$TM = [AmosEngineLib.AmosEngine+TMatrixID]

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
$names=@("PC_NT_RI","FS_NT_RI","PARV_NT_RI","PFE_ST_RI","WCE_ST_RI","PSC_ST_RI","PARV_ST_RI")
$ants =@("PC","FS","PARV","PFE","WCE","PSC","PARV")
$meds =@("NT","NT","NT","ST","ST","ST","ST")
$inv=[Globalization.CultureInfo]::InvariantCulture

function Run-BootModel($tag,$struct,$marker){
  if(Test-Path $Scratch){try{Remove-Item $Scratch -Force -ErrorAction Stop}catch{Start-Sleep -Milliseconds 600;try{Remove-Item $Scratch -Force -ErrorAction SilentlyContinue}catch{}}}
  $sem=New-Object AmosEngineLib.AmosEngine
  $L=New-Object System.Collections.Generic.List[string]
  foreach($x in $measure){$L.Add($x)}
  foreach($x in $struct){$L.Add($x)}
  foreach($v in @("PC","FS","PFE","WCE","PSC","PARV")){$L.Add("$v (v$v)")}
  $L.Add("zNT (vzNT)");$L.Add("zST (vzST)");$L.Add("zRI (vzRI)");$L.Add("zNT <> zST")
  for($i=1;$i -le 36;$i++){$L.Add("e$i (ve$i)")}
  $exo=@("PC","FS","PFE","WCE","PSC","PARV")
  for($a=0;$a -lt 6;$a++){for($b=$a+1;$b -lt 6;$b++){$L.Add("$($exo[$a]) <> $($exo[$b])")}}
  $sem.TextOutput($true);$sem.Standardized($true);$sem.TotalEffects($true)
  $sem.Seed($SEED);$sem.Bootstrap($NB);$sem.ConfidenceBC(0.95);$sem.ConfidencePC(0.95)
  $DE=$TM::DirectEffects
  $sem.NeedEstimates($DE);$sem.NeedBootSampleEstimates($DE)
  $sem.BeginGroup($sav)
  foreach($x in $L){$sem.AStructure($x)}
  $rc=$sem.FitModel()
  Write-Host ("[{0}] rc={1} chi2={2:F4} df={3} NPAR={4}" -f $tag,$rc,$sem.Cmin(),$sem.Df(),$sem.NumberOfParameters())
  [string[]]$rn=$null;[string[]]$cn=$null
  $sem.RowNames($DE,[ref]$rn);$sem.ColumnNames($DE,[ref]$cn)
  $ri=@{};for($i=0;$i -lt $rn.Length;$i++){$ri[$rn[$i]]=$i}
  $ci=@{};for($j=0;$j -lt $cn.Length;$j++){$ci[$cn[$j]]=$j}
  $bnt=$sem.GetEstimate($DE,"RI","NT");$bst=$sem.GetEstimate($DE,"RI","ST")
  "name,point" | Set-Content (Join-Path $Work "boot_specific_$tag.csv") -Encoding UTF8
  $prows=New-Object System.Collections.Generic.List[string]
  $srows=New-Object System.Collections.Generic.List[string]
  $srows.Add($names -join ",")
  for($k=0;$k -lt 7;$k++){
    $aa=$sem.GetEstimate($DE,$meds[$k],$ants[$k]);$bb=if($meds[$k] -eq "NT"){$bnt}else{$bst}
    $prows.Add(("{0},{1}" -f $names[$k],($aa*$bb).ToString("F6",$inv)))
  }
  $prows | Add-Content (Join-Path $Work "boot_specific_$tag.csv")
  $okc=0
  for($s=1;$s -le $NB;$s++){
    [double[,]]$bs=$null;$sem.GetBootSampleEstimates($DE,[ref]$bs,$s)
    if($bs -eq $null){continue};$okc++
    $xnt=$bs[$ri["RI"],$ci["NT"]];$xst=$bs[$ri["RI"],$ci["ST"]];$vals=@()
    for($k=0;$k -lt 7;$k++){
      $aa=$bs[$ri[$meds[$k]],$ci[$ants[$k]]];$bb=if($meds[$k] -eq "NT"){$xnt}else{$xst}
      $vals += ($aa*$bb).ToString("F6",$inv)
    }
    $srows.Add($vals -join ",")
  }
  $srows | Set-Content (Join-Path $Work "boot_samples_$tag.csv") -Encoding UTF8
  Write-Host "[$tag] valid bootstrap samples = $okc / $NB"
  $sem.Dispose()
  $done=$false
  for($k=0;$k -lt 40;$k++){Start-Sleep -Milliseconds 400;if(Test-Path $Scratch){$t=Get-Content $Scratch -Raw;if(($t -match $marker) -and ($t -match "Indirect")){$done=$true;break}}}
  if($done){Copy-Item $Scratch (Join-Path $OutDir "AMOS_Bootstrap_${tag}_原生输出报告.html") -Force}
}

# --- Model A: theoretical main model (9 structural paths) ---
$structA=@(
 "NT = (pc_nt) PC + (fs_nt) FS + (parv_nt) PARV + (1) zNT",
 "ST = (pfe_st) PFE + (wce_st) WCE + (psc_st) PSC + (parv_st) PARV + (1) zST",
 "RI = (nt_ri) NT + (st_ri) ST + (1) zRI")
Run-BootModel "modelA" $structA "1280\.32"

# --- Partial mediation: all 6 antecedents -> NT/ST and -> RI (20 paths) ---
$structP=@(
 "NT = (pc_nt) PC + (fs_nt) FS + (pfe_nt) PFE + (wce_nt) WCE + (psc_nt) PSC + (parv_nt) PARV + (1) zNT",
 "ST = (pc_st) PC + (fs_st) FS + (pfe_st) PFE + (wce_st) WCE + (psc_st) PSC + (parv_st) PARV + (1) zST",
 "RI = (pc_ri) PC + (fs_ri) FS + (pfe_ri) PFE + (wce_ri) WCE + (psc_ri) PSC + (parv_ri) PARV + (nt_ri) NT + (st_ri) ST + (1) zRI")
Run-BootModel "partial" $structP "1184\.16"

# copy per-sample files for reproducibility record
Copy-Item (Join-Path $Work "boot_samples_modelA.csv") (Join-Path $OutDir "Bootstrap5000_逐样本特定间接效应_模型A.csv") -Force
Copy-Item (Join-Path $Work "boot_samples_partial.csv") (Join-Path $OutDir "Bootstrap5000_逐样本特定间接效应_部分中介.csv") -Force
Write-Host "ALL DONE. Outputs in $OutDir"
