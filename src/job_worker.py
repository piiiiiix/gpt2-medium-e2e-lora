"""后台批次执行入口：在 lora 环境中顺序启动任务，保存进程日志与批次状态。"""

import subprocess
import sys

from config import OUTPUT_ROOT, PROJECT_ROOT, inside_project, require_lora
from utils.jobs import SCRIPTS, hidden_flags
from utils.runs import now, read_json, write_json


def main(batch):
    """顺序执行批次中的排队任务，记录失败并在结束时释放批次锁。"""
    require_lora()
    batch = inside_project(batch)
    metadata = read_json(batch)
    metadata.update(status="running", started=now())
    write_json(batch, metadata)
    failed = False
    try:
        for run in metadata["runs"]:
            path = inside_project(run["path"])
            with (path / "process.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(
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
                )
            status = read_json(path / "status.json", {})
            if result.returncode and status.get("status") not in ("failed", "interrupted"):
                write_json(
                    path / "status.json",
                    {
                        **status,
                        "status": "failed",
                        "ended": now(),
                        "error": f"Backend process exited {result.returncode}; see process.log",
                    },
                )
            failed = failed or result.returncode != 0
        metadata.update(status="failed" if failed else "completed", ended=now())
    finally:
        write_json(batch, metadata)
        lock = OUTPUT_ROOT / "jobs/active.lock"
        if read_json(lock, {}).get("batch") == str(batch):
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
