"""进程管理模块：启动独立任务批次、检查进程存活、准备对应训练运行的 TensorBoard。"""

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from config import OUTPUT_ROOT, PROJECT_ROOT, inside_project, require_lora
from utils.runs import new_run, now, read_json, write_json

SCRIPTS = {
    "lora": "train_lora.py",
    "fft": "train_fft.py",
    "tests": "test_e2e.py",
    "benchmarks": "run_benchmarks.py",
}


def alive(pid):
    """按进程 ID 检查后台任务是否仍存活，不向进程发送终止信号。"""
    if not pid:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def hidden_flags():
    """返回 Windows 隐藏子进程窗口的标志；其他系统返回 0。"""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def launch(tasks, existing_runs=None):
    """创建运行目录与批次锁，启动脱离前端生命周期的后台 worker。"""
    require_lora()
    folder = OUTPUT_ROOT / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "active.lock"
    if lock.exists():
        metadata = read_json(lock, {})
        if alive(metadata.get("pid")):
            raise RuntimeError("已有任务批次正在运行；请待其完成再启动，避免 GPU 任务竞争。")
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps(dict(pid=os.getpid(), starting=True)).encode())
    os.close(descriptor)
    try:
        if existing_runs is None:
            runs = [dict(kind=kind, path=str(new_run(kind, config))) for kind, config in tasks]
        else:
            runs = existing_runs
            for run, (_, config) in zip(runs, tasks):
                path = inside_project(run["path"])
                previous = read_json(path / "status.json")
                attempts = read_json(path / "attempts.json", [])
                attempts.append(
                    dict(
                        status=previous,
                        resumed_at=now(),
                        checkpoint=config["resume_from_checkpoint"],
                    )
                )
                write_json(path / "attempts.json", attempts)
                (path / "stop.request.json").unlink(missing_ok=True)
                write_json(path / "config.json", config)
                checkpoint_state = read_json(
                    Path(config["resume_from_checkpoint"]) / "trainer_state.json"
                )
                write_json(
                    path / "progress.json",
                    dict(
                        completed=checkpoint_state["global_step"],
                        total=checkpoint_state["max_steps"],
                        phase="resuming",
                    ),
                )
                write_json(
                    path / "status.json", dict(kind=run["kind"], status="queued", created=now())
                )
        batch = folder / (Path(runs[0]["path"]).name + "_" + uuid4().hex[:8] + ".json")
        write_json(batch, dict(status="queued", runs=runs, created=now()))
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-s", "-u", str(PROJECT_ROOT / "src/job_worker.py"), str(batch)],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=log,
                creationflags=hidden_flags(),
            )
        write_json(lock, dict(pid=proc.pid, batch=str(batch)))
        return runs
    except BaseException:
        lock.unlink(missing_ok=True)
        raise


def stop_batch():
    """请求当前 worker 停止整个批次；由 worker 停止子进程且取消后续排队项。"""
    require_lora()
    lock = read_json(OUTPUT_ROOT / "jobs/active.lock", {})
    if not lock.get("batch") or not alive(lock.get("pid")):
        raise ValueError("当前没有存活的后台批次。")
    batch = inside_project(lock["batch"])
    write_json(batch.with_suffix(".stop"), dict(requested=now(), reason="用户停止整个批次"))
    return batch


def terminate_tree(process):
    """停止 worker 自己创建的任务进程树，并等待文件写入进程退出。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            creationflags=hidden_flags(),
        )
        if result.returncode and process.poll() is None:
            raise RuntimeError("任务进程树未能停止，请查看后台日志。")
    else:
        os.killpg(process.pid, 15)
    process.wait(timeout=15)


def resume_training(run, checkpoint):
    """用原参数在原目录续训，恢复完整 Trainer 状态并保留上次运行状态记录。"""
    from utils.checkpoints import validate

    path = inside_project(run)
    status = read_json(path / "status.json", {})
    stopped = status.get("status") in ("interrupted", "failed") or (
        status.get("status") == "running" and not alive(status.get("pid"))
    )
    if status.get("kind") not in ("lora", "fft") or not stopped:
        raise ValueError("只能恢复已停止或失败的训练。")
    checkpoint = validate(path, checkpoint)
    state = read_json(checkpoint / "trainer_state.json")
    if state["global_step"] >= state["max_steps"]:
        raise ValueError("该断点已完成计划训练步数，没有剩余训练可恢复。")
    config = read_json(path / "config.json")
    config["resume_from_checkpoint"] = str(checkpoint)
    return launch([(status["kind"], config)], [dict(kind=status["kind"], path=str(path))])


def tensorboard(training_run):
    """复用或启动指定训练目录的 TensorBoard，等待就绪后返回本地地址。"""
    require_lora()
    path = inside_project(training_run)
    logdir = path / "tensorboard"
    if not list(logdir.glob("events.*")):
        raise ValueError("该训练运行没有 TensorBoard event 文件。")
    existing = read_json(path / "tensorboard_server.json", {})
    if alive(existing.get("pid")):
        return existing["url"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (path / "tensorboard_server.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-s",
                "-m",
                "tensorboard.main",
                "--logdir",
                str(logdir),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=log,
            creationflags=hidden_flags(),
        )
    metadata = dict(pid=proc.pid, url=f"http://127.0.0.1:{port}", logdir=str(logdir))
    write_json(path / "tensorboard_server.json", metadata)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("TensorBoard 启动失败，请查看 tensorboard_server.log。")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return metadata["url"]
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("TensorBoard 启动超时，请查看 tensorboard_server.log 后重试。")
