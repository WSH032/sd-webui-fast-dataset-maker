function InstallFail {
    Write-Output "��װʧ�ܡ�"
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
    Write-Output "���ڴ������⻷��..."
    python -m venv venv
    Check "�������⻷��ʧ�ܣ����� python �Ƿ�װ����Լ� python �汾�Ƿ�Ϊ64λ�汾��python 3.10����python��Ŀ¼�Ƿ��ڻ�������PATH�ڡ�"
}

.\venv\Scripts\activate
Check "�������⻷��ʧ�ܡ�"

Write-Output "��װ����"
pip install -r requirements.txt
Check "������װʧ�ܡ�"

Write-Output "��װ���"
Read-Host | Out-Null ;
