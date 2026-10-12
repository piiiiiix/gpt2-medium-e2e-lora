"""TensorBoard 启动入口：仅聚合 medium 正式训练的 event 日志，排除冒烟。"""

import subprocess
import sys
from pathlib import Path

from config import require_lora
from utils.runs import history


def logdir_spec():
    """为每个正式训练创建独立 TensorBoard 名称，避免扫描整个 outputs。"""
    logs = []
    for record in history():
        if record["kind"] not in ("lora", "fft"):
            continue
        path = Path(record["path"])
        directory = path / "tensorboard"
        if list(directory.glob("events.*")):
            logs.append(f"{record['kind']}_{path.name}:{directory}")
    return ",".join(logs)


def main():
    """使用 lora 环境启动本地 TensorBoard，透传端口等参数与进程返回码。"""
    require_lora()
    spec = logdir_spec()
    if not spec:
        raise SystemExit("没有找到正式训练的 TensorBoard event 日志。")
    command = [
        sys.executable,
        "-s",
        "-m",
        "tensorboard.main",
        "--logdir_spec",
        spec,
        "--host",
        "127.0.0.1",
        "--port",
        "6006",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
