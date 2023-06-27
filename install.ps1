function InstallFail {
    Write-Output "Install failed."
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating venv..."
    python -m venv venv
    Check "Creating venv failed, please check your python."
}

.\venv\Scripts\activate
Check "Activate venv failed."

"
If you want to use CUDA, you should choose Y.
"

$install_torch = Read-Host "Install torch==2.0.0 + CUDA==118 ? Y/N (default Y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq ""){
    Write-Output "Installing torch==2.0.0 + CUDA==118..."
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    Check "Install torch failed, please delet venv and run again."
    Write-Output "Install torch successfully."
    Start-Sleep -Seconds 1
}

Write-Output "pip requirements.txt installing..."
pip install -r requirements.txt
Check "pip requirements.txt install failed."

Write-Output "All done"
Read-Host | Out-Null ;
