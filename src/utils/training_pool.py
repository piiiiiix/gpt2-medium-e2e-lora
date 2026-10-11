"""待训练池：持久保存参数快照，并将选中的待训练项交给串行后台批次。"""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from config import OUTPUT_ROOT
from utils.jobs import launch
from utils.runs import now, read_json, write_json

POOL_ROOT = OUTPUT_ROOT / "training_pool"


def entries():
    """按添加时间读取池中项目，并从对应运行目录获取最新状态。"""
    result = []
    for path in POOL_ROOT.glob("*.json"):
        entry = read_json(path)
        if entry.get("run"):
            entry["status"] = read_json(
                OUTPUT_ROOT / entry["run"] / "status.json", {"status": "queued"}
            )["status"]
        result.append(entry)
    return sorted(result, key=lambda entry: (entry["created"], entry["id"]))


def add(kind, config, name=""):
    """保存当前参数的独立快照；后续表单修改不影响已入池的项目。"""
    if kind not in ("lora", "fft"):
        raise ValueError("待训练池仅接受 LoRA 或 FFT。")
    if kind == "lora" and not any(config["ranks"].values()):
        raise ValueError("全部投影已冻结，请启用至少一个 rank。")
    POOL_ROOT.mkdir(parents=True, exist_ok=True)
    entry = dict(
        id=uuid4().hex,
        name=name.strip() or f"{kind.upper()} · {now()}",
        kind=kind,
        config=deepcopy(config),
        created=now(),
        status="pending",
    )
    write_json(POOL_ROOT / f"{entry['id']}.json", entry)
    return entry


def start_pending(ids):
    """按入池顺序提交指定待训练项，成功启动后关联运行，防止重复提交。"""
    selected = [entry for entry in entries() if entry["id"] in ids]
    if not selected or len(selected) != len(set(ids)):
        raise ValueError("请选择存在的待训练项。")
    if any(entry["status"] != "pending" for entry in selected):
        raise ValueError("只能启动尚未提交的待训练项。")
    runs = launch([(entry["kind"], entry["config"]) for entry in selected])
    for entry, run in zip(selected, runs):
        entry.update(status="queued", run=str(Path(run["path"]).relative_to(OUTPUT_ROOT)))
        write_json(POOL_ROOT / f"{entry['id']}.json", entry)
    return runs


def remove(entry_id):
    """移除尚未提交的参数项；已提交任务及其训练记录保留。"""
    entry = next((entry for entry in entries() if entry["id"] == entry_id), None)
    if not entry or entry["status"] != "pending":
        raise ValueError("只能移除尚未提交的待训练项。")
    (POOL_ROOT / f"{entry_id}.json").unlink()
