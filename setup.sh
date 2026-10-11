#!/usr/bin/env bash
# 启动 medium 的 Streamlit 前端，所有 Python 依赖使用已有 conda lora 环境。
set -euo pipefail

# 根据脚本自身位置定位项目，允许从其他目录启动。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

# 优先使用当前 conda；兼容尚未初始化 conda 的本机 Git Bash。
if [[ -n "${CONDA_EXE:-}" && -f "$CONDA_EXE" ]]; then
    CONDA_COMMAND="$CONDA_EXE"
elif type -P conda >/dev/null 2>&1; then
    CONDA_COMMAND="$(type -P conda)"
elif [[ -x /c/ProgramData/anaconda3/Scripts/conda.exe ]]; then
    CONDA_COMMAND=/c/ProgramData/anaconda3/Scripts/conda.exe
else
    echo "找不到 conda，请将 conda 加入 PATH，或设置 CONDA_EXE 为 conda 可执行文件路径。" >&2
    exit 1
fi

# 不重复安装依赖；透传 Streamlit 参数，直接显示前端进程日志。
exec "$CONDA_COMMAND" run --no-capture-output -n lora \
    python -s -m streamlit run "$PROJECT_DIR/src/app.py" "$@"
