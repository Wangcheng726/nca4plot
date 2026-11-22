param(
  [Parameter(Mandatory=$true)][string]$Target,
  [Parameter(Mandatory=$true)][string]$Out,
  [int]$Iters = 8000,
  [int]$Pad = 0,
  [int]$CkptEvery = 500,
  [string]$Extra = ""
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"
$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$log = Join-Path $Out "train.log"
$errlog = Join-Path $Out "train.err.log"
$pyArgs = @("-m","nca4plot.train_plant","--target",$Target,"--iters","$Iters","--out",$Out,"--pad","$Pad","--ckpt-every","$CkptEvery")
if ($Extra -ne "") { $pyArgs += ($Extra -split " ") }
$p = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $errlog
$p.Id | Set-Content (Join-Path $Out "train.pid")
$p.WaitForExit()
"DONE exit=$($p.ExitCode)" | Add-Content (Join-Path $Out "runner.log")
