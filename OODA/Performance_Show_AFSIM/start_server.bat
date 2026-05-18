@echo off
chcp 65001 >nul
title 集群网络仿真平台 - 前端服务器
echo ========================================
echo   集群网络仿真平台综合态势 - 前端服务器
echo ========================================
echo.
cd /d "%~dp0"
echo 工作目录: %cd%
echo 浏览器访问: http://localhost:5500
echo 按 Ctrl+C 停止服务器
echo.
python --version >nul 2>&1
if %errorlevel% equ 0 (
    start http://localhost:5500
    python -m http.server 5500
    goto :end
)
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    start http://localhost:5500
    python3 -m http.server 5500
    goto :end
)
echo [启动方式] PowerShell HTTP Server
start http://localhost:5500
powershell -NoProfile -ExecutionPolicy Bypass -Command "$listener = New-Object System.Net.HttpListener; $listener.Prefixes.Add('http://localhost:5500/'); $listener.Start(); Write-Host 'Server started on port 5500'; while ($listener.IsListening) { $ctx = $listener.GetContext(); $path = $ctx.Request.Url.LocalPath; if ($path -eq '/') { $path = '/index.html' }; $file = Join-Path '%cd%' ($path -replace '/','\'); if (Test-Path $file -PathType Leaf) { $bytes = [IO.File]::ReadAllBytes($file); $ext = [IO.Path]::GetExtension($file); $mime = @{'.html'='text/html;charset=utf-8';'.css'='text/css';'.js'='application/javascript';'.json'='application/json';'.png'='image/png';'.jpg'='image/jpeg';'.svg'='image/svg+xml';'.ttf'='font/ttf'}; $ctx.Response.ContentType = if($mime[$ext]){$mime[$ext]}else{'application/octet-stream'}; $ctx.Response.ContentLength64 = $bytes.Length; $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length) } else { $ctx.Response.StatusCode = 404; $msg = [Text.Encoding]::UTF8.GetBytes('404'); $ctx.Response.OutputStream.Write($msg, 0, $msg.Length) }; $ctx.Response.Close() }"
:end
pause
