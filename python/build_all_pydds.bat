@echo off
setlocal enabledelayedexpansion

REM ===================== 配置多个 Python 版本路径 =====================
REM 修改为你机器上实际安装的路径
set PYTHON_VERSIONS=^
"C:\Users\AeroZeng\AppData\Local\Programs\Python\Python311\python.exe" ^
"C:\Users\AeroZeng\AppData\Local\Programs\Python\Python312\python.exe" ^
"C:\Users\AeroZeng\AppData\Local\Programs\Python\Python313\python.exe"

REM 项目路径（setup.py 所在目录）
set PROJECT_DIR=%~dp0

REM 输出目录（用于集中保存 .pyd 文件）
set DIST_DIR=%PROJECT_DIR%dist
if not exist %DIST_DIR% (
    mkdir %DIST_DIR%
)

echo [INFO] 项目路径：%PROJECT_DIR%
echo [INFO] 输出路径：%DIST_DIR%
echo.

REM ===================== 开始构建循环 =====================
for %%P in (%PYTHON_VERSIONS%) do (
    echo [INFO] 构建使用 Python：%%P

    REM 清理之前的构建结果
    if exist build (
        rmdir /s /q build
    )

    pushd %PROJECT_DIR%
    call "%%P" setup.py build_ext --inplace
    if errorlevel 1 (
        echo [ERROR] 构建失败：%%P
        exit /b 1
    )

    REM 获取生成的 .pyd 文件名
    for /f "delims=" %%F in ('dir /b /s /a:-d pydds\*.pyd') do (
        echo [INFO] 构建成功，结果：%%~nxF
        copy /Y "%%F" "%DIST_DIR%\%%~nxF" >nul
    )
    popd

    echo.
)

echo [SUCCESS] 所有版本构建完成，输出文件保存在：%DIST_DIR%
pause
