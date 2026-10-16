"""统一任务列表的状态映射、组合筛选和创建时间排序。"""

import re
from datetime import datetime
from pathlib import Path

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


def parse_time(value):
    """兼容时区、UTC Z 及不同小数位数，统一为香港时间。"""
    try:
        value = re.sub(
            r"(?<=\d{2}:\d{2}:\d{2})\.(\d+)",
            lambda match: "." + match.group(1)[:6].ljust(6, "0"),
            value,
        )
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=HK)).astimezone(HK)
    except (ValueError, TypeError, AttributeError):
        return None


def task_time(record):
    """恢复历史任务时间，返回时间及来源，不修改磁盘记录。"""
    stamp = parse_time(record.get("created"))
    if stamp:
        return stamp, "创建"
    match = re.match(r"^(\d{8}_\d{6}_\d{6})(?:_|$)", Path(record.get("path", "")).name)
    if match:
        try:
            return datetime.strptime(match[1], "%Y%m%d_%H%M%S_%f").replace(
                tzinfo=HK
            ), "创建（目录）"
        except ValueError:
            pass
    for key, label in (("first_started", "首次开始"), ("started", "开始"), ("ended", "结束")):
        stamp = parse_time(record.get(key))
        if stamp:
            return stamp, label
    return None, "时间未知"


def time_label(record):
    """展示带明确来源的香港时间，与排序使用相同时间。"""
    stamp, source = task_time(record)
    return f"{source}：{stamp:%Y-%m-%d %H:%M:%S}（香港时间）" if stamp else source


def created_time(record):
    """按恢复的任务时间跨目录排序，完全缺失时排末尾。"""
    stamp, _ = task_time(record)
    return stamp.timestamp() if stamp else float("-inf")


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
