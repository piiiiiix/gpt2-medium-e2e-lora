"""Fetch checksum-pinned paper data and the unmodified official E2E evaluator."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from config import OUTPUT_ROOT

MANIFEST = Path(__file__).with_name("paper_sources.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(item, destination, offline=False):
    destination = Path(destination)
    if destination.exists() and sha256(destination) == item["sha256"]:
        return destination
    if offline:
        raise FileNotFoundError(
            f"Missing/changed paper resource: {destination}; run prepare_paper.py"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    try:
        with (
            os.fdopen(fd, "wb") as stream,
            urllib.request.urlopen(item["url"], timeout=120) as response,
        ):
            shutil.copyfileobj(response, stream)
        actual = sha256(name)
        if actual != item["sha256"]:
            raise ValueError(
                f"Checksum mismatch: {item['url']}; "
                f"expected {item['sha256']}, got {actual} "
                f"({Path(name).stat().st_size} bytes)"
            )
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return destination


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def data_files(offline=False):
    return {
        split: fetch(item, OUTPUT_ROOT / "resources/paper_e2e" / f"{split}.txt", offline)
        for split, item in manifest()["data"].items()
    }


def evaluator(offline=False):
    source = manifest()["evaluator"]
    root = OUTPUT_ROOT / "resources" / f"e2e-metrics-{source['commit']}"
    if all(
        (root / name).is_file() and sha256(root / name) == digest
        for name, digest in source["files"].items()
    ):
        return root
    archive = fetch(source, root.with_suffix(".zip"), offline)
    with zipfile.ZipFile(archive) as zipped:
        for name, digest in source["files"].items():
            payload = zipped.read(f"e2e-metrics-{source['commit']}/{name}")
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Evaluator file mismatch: {name}")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return root


def runtime_env():
    """Use PATH installations or project-local Java/Perl; never silently change scorer."""
    env = os.environ.copy()
    folders = []
    for variable, executable in (("JAVA_HOME", "java.exe"), ("PERL_HOME", "perl.exe")):
        if env.get(variable):
            folders.append(str(Path(env[variable]) / "bin"))
        folders.extend(
            str(p.parent) for p in (OUTPUT_ROOT / "resources/runtime").glob(f"**/{executable}")
        )
    env["PATH"] = os.pathsep.join(folders + [env.get("PATH", "")])
    env["PYTHONUTF8"] = "1"
    local_python = OUTPUT_ROOT / "resources/paper_python"
    if local_python.is_dir():
        env["PYTHONPATH"] = str(local_python) + os.pathsep + env.get("PYTHONPATH", "")
    for command in ("java", "perl"):
        if not shutil.which(command, path=env["PATH"]):
            raise RuntimeError(f"Official E2E scoring requires {command} on PATH (see README).")
    return env


def preflight(offline=False):
    root, env = evaluator(offline), runtime_env()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import measure_scores",
            str(root),
        ],
        env=env,
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            "Official scorer dependencies unavailable; install requirements.txt.\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return root, env
