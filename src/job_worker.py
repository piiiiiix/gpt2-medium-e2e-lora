"""后台批次执行入口：在 lora 环境中顺序启动任务，保存进程日志与批次状态。"""

import os
import subprocess
import sys
import time

from config import OUTPUT_ROOT, PROJECT_ROOT, inside_project, require_lora
from utils.jobs import SCRIPTS, hidden_flags, terminate_tree
from utils.runs import now, read_json, write_json


def main(batch):
    """顺序执行批次中的排队任务，记录失败并在结束时释放批次锁。"""
    require_lora()
    batch = inside_project(batch)
    metadata = read_json(batch)
    metadata.update(status="running", started=now())
    write_json(batch, metadata)
    failed = False
    stopped = False
    try:
        for run in metadata["runs"]:
            if batch.with_suffix(".stop").exists():
                stopped = True
                break
            path = inside_project(run["path"])
            with (path / "process.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-s",
                        "-u",
                        str(PROJECT_ROOT / "src" / SCRIPTS[run["kind"]]),
                        "--config",
                        str(path / "config.json"),
                        "--run-dir",
                        str(path),
                    ],
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=log,
                    creationflags=hidden_flags(),
                    start_new_session=os.name != "nt",
                )
                metadata.update(current_pid=process.pid, current_run=str(path))
                write_json(batch, metadata)
                stop_deadline = None
                while process.poll() is None:
                    if batch.with_suffix(".stop").exists():
                        stopped = True
                        if run["kind"] not in ("lora", "fft"):
                            terminate_tree(process)
                            break
                        if stop_deadline is None:
                            write_json(path / "stop.request.json", dict(requested=now()))
                            stop_deadline = time.monotonic() + 90
                        # 先在 step 边界保存断点，超时则停止整个任务进程树。
                        if time.monotonic() >= stop_deadline:
                            terminate_tree(process)
                            break
                    time.sleep(0.2)
                returncode = process.wait()
            if stopped:
                break
            status = read_json(path / "status.json", {})
            if returncode and status.get("status") not in ("failed", "interrupted"):
                write_json(
                    path / "status.json",
                    {
                        **status,
                        "status": "failed",
                        "ended": now(),
                        "error": f"Backend process exited {returncode}; see process.log",
                    },
                )
            failed = failed or returncode != 0
        if stopped:
            for run in metadata["runs"]:
                path = inside_project(run["path"])
                state = read_json(path / "status.json", {})
                if state.get("status") in ("queued", "running"):
                    write_json(
                        path / "status.json",
                        {
                            **state,
                            "status": "interrupted",
                            "ended": now(),
                            "error": "用户停止整个批次；仅可从已完整保存的 checkpoint 续训。",
                        },
                    )
        metadata.update(
            status="interrupted" if stopped else ("failed" if failed else "completed"), ended=now()
        )
    finally:
        write_json(batch, metadata)
        lock = OUTPUT_ROOT / "jobs/active.lock"
        if read_json(lock, {}).get("batch") == str(batch):
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
