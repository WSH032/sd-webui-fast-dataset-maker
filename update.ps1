# 提示信息
"
#############################################

Pulling the latest code from the remote repository
There may be changes in the dependent environment, it is recommended to run 'install.ps1' again

#############################################

"

git reset --hard
git pull
git submodule init
git submodule update
pause
