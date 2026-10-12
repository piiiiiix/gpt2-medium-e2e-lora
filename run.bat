@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
rem 启动 medium 前端，使用已有 conda lora 环境，不自动安装依赖。

rem 根据脚本位置定位项目，允许从其他目录或双击启动。
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem 优先使用 CONDA_EXE，再查找 PATH，最后兼容本机 Anaconda 安装路径。
set "CONDA_COMMAND="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_COMMAND=%CONDA_EXE%"
if not defined CONDA_COMMAND for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_COMMAND set "CONDA_COMMAND=%%I"
if not defined CONDA_COMMAND for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA_COMMAND set "CONDA_COMMAND=%%I"
if not defined CONDA_COMMAND if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_COMMAND=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_COMMAND goto :missing_conda

rem 禁用 Python 用户包，透传所有 Streamlit 参数，并保留进程返回码。
call "%CONDA_COMMAND%" run --no-capture-output -n lora python -s -m streamlit run "%~dp0src\app.py" %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %RUN_EXIT_CODE%

:missing_conda
echo 找不到 conda，请将 conda 加入 PATH，或设置 CONDA_EXE 为 conda 可执行文件路径。 1>&2
popd
endlocal & exit /b 1
