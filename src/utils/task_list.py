"""统一任务列表的状态映射、组合筛选和创建时间排序。"""

import re
from datetime import datetime

from utils.jobs import alive
from utils.runs import HK


def task_status(record):
    """将中断、异常退出统一归入失败，原始记录保持不变。"""
    state = record.get("status")
    if state == "running" and not alive(record.get("pid")):
        return "失败"
    return {
        "completed": "已完成",
        "running": "运行中",
        "queued": "队列中",
        "failed": "失败",
        "interrupted": "失败",
    }.get(state, "失败")


def created_time(record):
    """按真实创建时间跨目录排序，缺失或损坏时间排在末尾。"""
    try:
        value = re.sub(
            r"(?<=\d{2}:\d{2}:\d{2})\.(\d+)",
            lambda match: "." + match.group(1)[:6].ljust(6, "0"),
            record["created"],
        )
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=HK)).timestamp()
    except (KeyError, ValueError, TypeError, AttributeError):
        return float("-inf")


def filter_tasks(records, kind="全部", model="全部", status="全部"):
    """将 tests 与 benchmarks 合并展示，三个维度以 AND 组合。"""
    items = []
    for record in records:
        training = record["kind"] in ("lora", "fft")
        model_type = record["kind"] if training else record["config"].get("model", {}).get("type")
        if kind != "全部" and kind != ("训练" if training else "benchmark"):
            continue
        if model != "全部" and model.lower() != model_type:
            continue
        if status != "全部" and status != task_status(record):
            continue
        items.append(record)
    return sorted(items, key=lambda record: (created_time(record), record["path"]), reverse=True)
