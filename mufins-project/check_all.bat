@echo off

call conda activate venv_mufins\ || pause && exit /b

echo #########################################
echo mypy
echo ..checking mufins
call python -m mypy mufins || pause && exit /b
for /f %%F in ('dir bin\*.py /s /b') do (
    echo ..checking %%F
    call python -m mypy %%F || pause && exit /b
)
for /f %%F in ('dir tools\*.py /s /b') do (
    echo ..checking %%F
    call python -m mypy %%F || pause && exit /b
)
for /f %%F in ('dir *.py /b') do (
    echo ..checking %%F
    call python -m mypy %%F || pause && exit /b
)
echo.

echo #########################################
echo pylint
echo ..checking mufins
call python -m pylint mufins || pause && exit /b
for /f %%F in ('dir bin\*.py /s /b') do (
    echo ..checking %%F
    call python -m pylint %%F || pause && exit /b
)
for /f %%F in ('dir tools\*.py /s /b') do (
    echo ..checking %%F
    call python -m pylint %%F || pause && exit /b
)
for /f %%F in ('dir *.py /b') do (
    echo ..checking %%F
    call python -m pylint %%F || pause && exit /b
)
echo.

echo #########################################
echo project validation
call python tools\validate_project.py || pause && exit /b
echo.

echo #########################################
echo sphinx
cd docs
call make html || cd .. && pause && exit /b
echo.

echo #########################################
echo unittest
cd ..\mufins\tests
call python -m unittest || cd ..\.. && pause && exit /b
echo.

cd ..\..
