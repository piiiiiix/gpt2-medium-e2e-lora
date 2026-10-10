"""Detached serialized task batches; frontend reruns do not own backend lifecycle."""
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from config import OUTPUT_ROOT, PROJECT_ROOT, inside_project, require_lora
from utils.runs import new_run, write_json, read_json, now

SCRIPTS={'lora':'train_lora.py','fft':'train_fft.py','tests':'test_e2e.py','benchmarks':'run_benchmarks.py'}

def alive(pid):
    if not pid:
        return False
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            return False
        code=ctypes.c_ulong()
        ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok and code.value==259)
    try:
        os.kill(pid,0)
        return True
    except OSError:
        return False

def hidden_flags():
    return getattr(subprocess,'CREATE_NO_WINDOW',0)

def launch(tasks):
    require_lora()
    folder=OUTPUT_ROOT/'jobs'
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'active.lock'
    if lock.exists():
        metadata=read_json(lock,{})
        if alive(metadata.get('pid')):
            raise RuntimeError('已有任务批次正在运行；请待其完成再启动，避免 GPU 任务竞争。')
        lock.unlink()
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,json.dumps(dict(pid=os.getpid(),starting=True)).encode())
    os.close(descriptor)
    try:
        runs=[dict(kind=kind,path=str(new_run(kind,config))) for kind,config in tasks]
        batch=folder/(Path(runs[0]['path']).name+'.json')
        write_json(batch,dict(status='queued',runs=runs,created=now()))
        with (folder/'worker.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-s','-u',str(PROJECT_ROOT/'src/job_worker.py'),str(batch)],
                cwd=PROJECT_ROOT,stdout=log,stderr=log,creationflags=hidden_flags())
        write_json(lock,dict(pid=proc.pid,batch=str(batch)))
        return runs
    except BaseException:
        lock.unlink(missing_ok=True)
        raise

def tensorboard(training_run):
    require_lora()
    path=inside_project(training_run)
    logdir=path/'tensorboard'
    if not list(logdir.glob('events.*')):
        raise ValueError('该训练运行没有 TensorBoard event 文件。')
    existing=read_json(path/'tensorboard_server.json',{})
    if alive(existing.get('pid')):
        return existing['url']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    with (path/'tensorboard_server.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-s','-m','tensorboard.main','--logdir',str(logdir),
            '--host','127.0.0.1','--port',str(port)],cwd=PROJECT_ROOT,
            stdout=log,stderr=log,creationflags=hidden_flags())
    metadata=dict(pid=proc.pid,url=f'http://127.0.0.1:{port}',logdir=str(logdir))
    write_json(path/'tensorboard_server.json',metadata)
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            raise RuntimeError('TensorBoard 启动失败，请查看 tensorboard_server.log。')
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=.2):
                return metadata['url']
        except OSError:
            time.sleep(.1)
    raise RuntimeError('TensorBoard 启动超时，请查看 tensorboard_server.log 后重试。')
