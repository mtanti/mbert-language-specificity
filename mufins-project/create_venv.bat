@echo off

call conda create --prefix venv_mufins\ python=3.6 || pause && exit /b
call conda activate venv_mufins\ || pause && exit /b

call pip install --upgrade pip || pause && exit /b
call pip install -r requirements.txt || pause && exit /b
call pip install -e . || pause && exit /b

call check_all.bat || pause && exit /b
