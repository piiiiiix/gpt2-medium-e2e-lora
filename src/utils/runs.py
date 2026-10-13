"""运行记录模块：负责目录命名、原子保存、日志、任务状态、历史查询与耗时估算。"""

import hashlib
import json
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import OUTPUT_ROOT, inside_project, require_lora
from utils.gpu_monitor import GPUMonitor

HK = timezone(timedelta(hours=8))


def now():
    """返回香港时区的 ISO 时间字符串，用于记录任务时间。"""
    return datetime.now(HK).isoformat()


def elapsed_between(started, ended):
    """兼容 PowerShell 七位小数与 UTC Z 时间；损坏或倒序时间返回不可估算。"""
    try:
        values = []
        for value in (started, ended):
            # Python 3.10 的 ISO 解析器最多支持六位微秒；只截断秒的小数部分。
            normalized = re.sub(
                r"(?<=\d{2}:\d{2}:\d{2})\.(\d{6})\d+",
                r".\1",
                value.strip(),
            )
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            stamp = datetime.fromisoformat(normalized)
            values.append(stamp if stamp.tzinfo else stamp.replace(tzinfo=HK))
        seconds = (values[1] - values[0]).total_seconds()
        return seconds if seconds >= 0 else None
    except (ValueError, TypeError, AttributeError):
        return None


def write_json(path, value):
    """先写临时文件再原子替换 JSON，避免前端读到半写入的状态。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path, default=None):
    """读取 JSON；文件不存在或暂时无法解析时返回默认值。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def digest(config):
    """生成配置的稳定短摘要，用于目录命名和运行配置识别。"""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]


def new_run(kind, config):
    """创建独立运行目录，将关键参数与时间戳写入名称，并保存初始配置和状态。"""
    import re

    training = kind in ("lora", "fft")
    folder = {"lora": "gpt2-e2e-lora", "fft": "gpt2-e2e-full-ft"}.get(kind, kind)
    if training:
        tag = (
            f"lr{config['learning_rate']:g}_e{config['epochs']}_b{config['batch_size']}"
            f"_ga{config['gradient_accumulation']}_s{config['seed']}"
        )
        if kind == "lora":
            ranks = config["ranks"]
            tag += (
                f"_q{ranks['q']}k{ranks['k']}v{ranks['v']}"
                if isinstance(ranks, dict)
                else "_legacyqkv" + digest(ranks)
            )
    else:
        tag = (
            f"{config['model']['type']}_{digest(config['model'])}_{config['split']}"
            f"_n{config['samples'] or 'all'}_t{config['max_new_tokens']}"
            f"_beam{config['num_beams']}_" + "-".join(config["metrics"])
        )
    tag = re.sub(r"[^a-zA-Z0-9_.-]", "-", tag)
    path = (
        (OUTPUT_ROOT / "smoke" if config.get("smoke") else OUTPUT_ROOT)
        / folder
        / (datetime.now(HK).strftime("%Y%m%d_%H%M%S_%f") + "_" + tag + "_" + digest(config))
    )
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", config)
    write_json(path / "status.json", dict(kind=kind, status="queued", created=now()))
    return path


class Tee:
    """标准输出分流器：同时写入控制台和运行日志，保留原始任务输出。"""

    def __init__(self, console, file):
        """绑定控制台流与本次运行的日志文件。"""
        self.console, self.file = console, file

    def write(self, text):
        """同时写入两个输出目标，并及时刷新日志文件。"""
        self.console.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        """刷新控制台和日志，确保当前输出已落盘。"""
        self.console.flush()
        self.file.flush()

    def isatty(self):
        """声明当前分流输出不是交互终端，避免进度条使用终端控制模式。"""
        return False


@contextmanager
def running(path, kind, config):
    """任务生命周期上下文：保存环境与状态、启动监控、捕获异常并记录总耗时。"""
    require_lora()
    path = inside_project(path)
    write_json(path / "config.json", config)
    metadata = dict(
        kind=kind,
        status="running",
        started=now(),
        pid=os.getpid(),
        config_hash=digest(config),
        hardware=GPUMonitor.sample(),
        python=sys.executable,
    )
    attempts = read_json(path / "attempts.json", [])
    metadata["first_started"] = next(
        (attempt["status"]["started"] for attempt in attempts if attempt["status"].get("started")),
        metadata["started"],
    )
    previous_seconds = 0
    for attempt in attempts:
        state = attempt["status"]
        seconds = state.get("elapsed_seconds")
        if seconds is None and state.get("started") and state.get("ended"):
            seconds = elapsed_between(state["started"], state["ended"])
            if seconds is None:
                metadata.setdefault("timing_warnings", []).append(
                    "历史时间记录无法计算耗时，已保留原记录并跳过该次耗时统计。"
                )
        previous_seconds += seconds or 0
    write_json(path / "status.json", metadata)
    import importlib.metadata

    versions = {}
    for package in [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "streamlit",
        "tensorboard",
        "nltk",
        "pycocoevalcap",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(path / "environment.json", versions)
    monitor = GPUMonitor(path / "gpu.jsonl")
    started = time.perf_counter()
    with (path / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            monitor.start()
            try:
                print(json.dumps(config, ensure_ascii=False, default=str), flush=True)
                yield
                metadata["status"] = "completed"
            except BaseException as exc:
                metadata.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=str(exc),
                )
                traceback.print_exc()
                raise
            finally:
                monitor.stop()
                metadata.update(ended=now(), elapsed_seconds=time.perf_counter() - started)
                metadata["total_elapsed_seconds"] = previous_seconds + metadata["elapsed_seconds"]
                write_json(path / "status.json", metadata)


def history(include_smoke=False):
    """读取正式运行状态及配置并排序；内部测试可显式包含冒烟记录。"""
    records = []
    if OUTPUT_ROOT.exists():
        for path in OUTPUT_ROOT.rglob("status.json"):
            if not include_smoke and "smoke" in path.relative_to(OUTPUT_ROOT).parts:
                continue
            try:
                inside_project(path)
            except ValueError:
                continue
            status = read_json(path)
            config = read_json(path.parent / "config.json", {})
            if config.get("smoke") and not include_smoke:
                continue
            if status:
                records.append(
                    dict(
                        path=str(path.parent),
                        **status,
                        config=config,
                    )
                )
    return sorted(records, key=lambda x: x["path"], reverse=True)


def estimate(config, kind):
    """用同配置、同 GPU 的成功正式运行估算均值，排除冒烟和缺失结果的记录。"""
    keys = [
        "model",
        "dataset",
        "local_data",
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "max_steps",
        "train_samples",
        "eval_samples",
        "precision",
        "gradient_checkpointing",
        "max_length",
        "ranks",
        "split",
        "samples",
        "max_new_tokens",
        "num_beams",
        "metrics",
        "predictions",
    ]
    signature = {k: config.get(k) for k in keys}
    current_gpu = [r.get("uuid") for r in GPUMonitor.sample()]
    durations = [
        r["elapsed_seconds"]
        for r in history()
        if r["status"] == "completed"
        and r["kind"] == kind
        and not r["config"].get("smoke")
        and (
            Path(r["path"]) / ("result.json" if kind in ("lora", "fft") else "metrics.json")
        ).exists()
        and {k: r["config"].get(k) for k in keys} == signature
        and [g.get("uuid") for g in r.get("hardware", [])] == current_gpu
    ]
    return (sum(durations) / len(durations), len(durations)) if durations else (None, 0)
