"""命令行配置模块：合并默认值与 JSON 覆盖参数，创建或接收排队中的运行目录。"""

import argparse
import copy

from config import inside_project
from utils.runs import new_run, read_json


def arguments(defaults, kind):
    """读取命令行参数、合并 JSON 配置并确定运行目录，拒绝覆盖已有任务。"""
    parser = argparse.ArgumentParser(description=f"GPT-2 Medium {kind}")
    parser.add_argument("--config", help="JSON overrides, path inside medium project")
    parser.add_argument("--run-dir", help="Existing queued run directory")
    args = parser.parse_args()
    config = copy.deepcopy(defaults)
    if args.config:
        overrides = read_json(inside_project(args.config))
        if not isinstance(overrides, dict):
            raise ValueError("Config must contain a JSON object")
        unknown = set(overrides) - set(config) - {"resume_from_checkpoint"}
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        config.update(overrides)
    path = inside_project(args.run_dir) if args.run_dir else new_run(kind, config)
    if args.run_dir and read_json(path / "status.json", {}).get("status") != "queued":
        raise ValueError("Only queued run directories may be started; never overwrite a prior run")
    return config, path
