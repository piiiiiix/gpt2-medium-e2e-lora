"""逐步冒烟测试入口：验证运行管理、模型、训练、评测、前端和异常边界。

微型模型用于快速检查，medium-* 使用真实 GPT-2 Medium；结果均标记为冒烟。"""

import argparse
import json
import time
from pathlib import Path

from config import TRAIN_DEFAULTS, require_lora
from utils.runs import new_run, read_json, running


def runtime():
    """验证成功与失败任务的状态、耗时、标准输出及 GPU 日志能够持久保存。"""
    config = {**TRAIN_DEFAULTS, "smoke": True}
    path = new_run("lora", config)
    with running(path, "lora", config):
        print("runtime smoke passed")
        time.sleep(0.1)
    assert read_json(path / "status.json")["status"] == "completed"
    assert json.loads((path / "gpu.jsonl").read_text().splitlines()[0])
    failed = new_run("fft", config)
    try:
        with running(failed, "fft", config):
            raise RuntimeError("deliberate smoke failure")
    except RuntimeError:
        pass
    assert read_json(failed / "status.json")["status"] == "failed"
    print("PASS: runtime success/failure status, log, GPU telemetry")


def model_pipeline():
    """验证独立 QKV rank、冻结、反向传播、adapter 重载及旧格式兼容。"""
    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    from config import OUTPUT_ROOT
    from utils.data import tokenize
    from utils.models import discover
    from utils.qkv_lora import attach, load_adapter, save_adapter
    from utils.runs import write_json

    folder = OUTPUT_ROOT / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tiny = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_layer=2,
            n_head=2,
            n_embd=32,
            n_positions=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    )
    tiny.save_pretrained(folder / "tiny-base")
    tokenizer.save_pretrained(folder / "tiny-base")
    rows = [
        dict(
            meaning_representation=f"name[Restaurant {i}], food[Italian]",
            target=f"Restaurant {i} serves Italian food.",
            references=[f"Restaurant {i} serves Italian food."],
        )
        for i in range(4)
    ]
    write_json(folder / "data.json", dict(train=rows, validation=rows, test=rows))
    assert any(x != -100 for x in tokenize(rows[0], tokenizer, 128)["labels"])
    ranks = dict(q=2, k=0, v=1)
    model = attach(tiny, ranks, alpha=32, dropout=0)
    assert "k" not in model.transformer.h[0].attn.c_attn.branches
    assert all(
        layer.attn.c_attn.branches["q"].a.out_features == 2
        and layer.attn.c_attn.branches["v"].a.out_features == 1
        and "k" not in layer.attn.c_attn.branches
        for layer in model.transformer.h
    )
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".branches." not in n)
    ids = torch.tensor([[1, 2, 3, 4]])
    model(ids, labels=ids).loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if ".branches." not in n)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".branches." in n:
                p.add_(0.01)
    model.eval()
    expected = model(ids).logits.detach()
    save_adapter(
        model,
        folder / "adapter",
        dict(model=str(folder / "tiny-base"), ranks=ranks, alpha=32, dropout=0),
    )
    reloaded = load_adapter(
        GPT2LMHeadModel.from_pretrained(folder / "tiny-base"), folder / "adapter"
    ).eval()
    torch.testing.assert_close(reloaded(ids).logits, expected)
    save_adapter(
        model,
        folder / "legacy-adapter",
        dict(
            model=str(folder / "tiny-base"),
            ranks=[ranks.copy() for _ in range(2)],
            alpha=32,
            dropout=0,
        ),
    )
    legacy = load_adapter(
        GPT2LMHeadModel.from_pretrained(folder / "tiny-base"), folder / "legacy-adapter"
    ).eval()
    torch.testing.assert_close(legacy(ids).logits, expected)
    k_only = attach(
        GPT2LMHeadModel.from_pretrained(folder / "tiny-base"), dict(q=0, k=3, v=0), dropout=0
    )
    assert all(
        set(layer.attn.c_attn.branches) == {"k"}
        and layer.attn.c_attn.branches["k"].a.out_features == 3
        for layer in k_only.transformer.h
    )
    k_only(ids, labels=ids).loss.backward()
    assert all(
        layer.attn.c_attn.branches["k"].b.weight.grad is not None for layer in k_only.transformer.h
    )
    assert any(m["path"] == str(folder / "adapter") for m in discover(include_smoke=True))
    print(
        "PASS: tokenization, model-wide independent QKV ranks, "
        "K freezing, backward, adapter save/reload, discovery"
    )


def training_pipeline(kind):
    """用微型模型验证指定训练方式的更新、验证、checkpoint 和最终生成。"""
    from copy import deepcopy

    from config import OUTPUT_ROOT
    from utils.models import load_model
    from utils.runs import write_json
    from utils.training import train

    config = deepcopy(TRAIN_DEFAULTS)
    config.update(
        model=str(OUTPUT_ROOT / "smoke/tiny-base"),
        local_data=str(OUTPUT_ROOT / "smoke/data.json"),
        max_steps=2,
        batch_size=2,
        eval_batch_size=2,
        logging_steps=1,
        warmup_steps=0,
        max_length=128,
        epochs=1,
        local_files_only=True,
        smoke=True,
        ranks=dict(q=2, k=0, v=1),
    )
    path = new_run(kind, config)
    train(config, path, kind)
    assert read_json(path / "status.json")["status"] == "completed"
    assert list((path / "tensorboard").glob("events.*"))
    assert read_json(path / "result.json")["best_checkpoint"]
    model, tokenizer = load_model(dict(type=kind, path=str(path / "final")), True, device="cpu")
    output = model.generate(
        **tokenizer("MR: name[Test]\nText:", return_tensors="pt"),
        max_new_tokens=2,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert output.shape[1] > 0
    write_json(OUTPUT_ROOT / f"smoke/{kind}-run.json", {"path": str(path)})
    print(
        f"PASS: {kind} training, evaluation, checkpoint, best reload, TensorBoard, final generation"
    )


def evaluation_pipeline():
    """验证三类模型的逐条预测、离线重评分一致性以及全部指标明细。"""
    from copy import deepcopy

    from config import EVAL_DEFAULTS, METRICS, OUTPUT_ROOT
    from utils.evaluation import evaluate
    from utils.scoring import read_predictions, score

    folder = OUTPUT_ROOT / "smoke"
    for kind in ["base", "lora", "fft"]:
        path = (
            str(folder / "tiny-base")
            if kind == "base"
            else read_json(folder / f"{kind}-run.json")["path"] + "/final"
        )
        config = deepcopy(EVAL_DEFAULTS)
        config.update(
            model=dict(type=kind, path=path),
            local_data=str(folder / "data.json"),
            samples=2,
            max_new_tokens=8,
            num_beams=2,
            local_files_only=True,
            metrics=["BLEU", "ROUGE-L", "CIDEr"],
            smoke=True,
        )
        run = new_run("benchmarks", config)
        evaluate(config, run)
        assert len(read_predictions(run / "predictions.jsonl")) == 2
        original = read_json(run / "metrics.json")["metrics"]
        config["predictions"] = str(run / "predictions.jsonl")
        repeat = new_run("benchmarks", config)
        evaluate(config, repeat)
        assert read_json(repeat / "metrics.json")["metrics"] == original
    fixture = folder / "scores-fixture.jsonl"
    with fixture.open("w", encoding="utf-8") as stream:
        for i in range(4):
            text = (
                f"Restaurant number {i} serves very delicious Italian food near the city centre ."
            )
            stream.write(
                json.dumps(dict(id=i, target=text, prediction=text, references=[text])) + "\n"
            )
    score(fixture, METRICS, folder)
    assert len(read_json(folder / "metrics.json")["metrics"]) == 8
    assert all("scores" in r for r in read_predictions(folder / "sample_scores.jsonl"))
    print(
        "PASS: Base/LoRA/FFT generation, incremental rows, "
        "offline rescoring, all five metrics and per-sample scores"
    )


def frontend_pipeline():
    """用 AppTest 验证前端控件与页面，并检查独立后台任务和 TensorBoard 接口。"""
    from copy import deepcopy
    from urllib.request import urlopen

    from streamlit.testing.v1 import AppTest

    from config import EVAL_DEFAULTS, OUTPUT_ROOT
    from utils.jobs import alive, launch, tensorboard
    from utils.runs import read_json

    existing_pids = {
        read_json(p, {}).get("pid") for p in OUTPUT_ROOT.rglob("tensorboard_server.json")
    }
    app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
    assert not app.exception
    learning_rate = app.number_input(key="learning_rate_lora_论文默认 / Prefix-Tuning FT 默认")
    assert learning_rate.proto.step == 1e-6
    lora_values = {item.label: item.value for item in app.number_input}
    assert lora_values["Seed"] == 110 and lora_values["验证 batch size"] == 4
    assert lora_values["LoRA dropout"] == 0.1
    assert not next(item for item in app.checkbox if item.label == "梯度检查点").value
    previous = learning_rate.value
    # 切换训练方式必须加载各自默认值，不能沿用 LoRA 的学习率。
    app.segmented_control(key="train_kind").set_value("fft").run()
    fft_values = {item.label: item.value for item in app.number_input}
    assert fft_values["学习率"] == 5e-5 and fft_values["每设备 batch size"] == 10
    assert fft_values["Seed"] == 101 and fft_values["最大训练长度"] == 1024
    assert not next(item for item in app.checkbox if item.label == "梯度检查点").value
    app.segmented_control(key="train_kind").set_value("lora").run()
    learning_rate = app.number_input(key="learning_rate_lora_论文默认 / Prefix-Tuning FT 默认")
    # 表单内先验证按钮产生的值；未提交表单时，后端仍保留旧值。
    learning_rate.increment()
    assert abs(learning_rate.value - previous - 1e-6) < 1e-12
    learning_rate.set_value(previous).run()
    app.number_input(key="q_rank").set_value(8).run()
    app.number_input(key="k_rank").set_value(2).run()
    app.number_input(key="v_rank").set_value(0).run()
    assert app.number_input(key="q_rank").value == 8 and app.number_input(key="k_rank").value == 2
    assert app.number_input(key="v_rank").value == 0
    assert not app.get("data_editor")
    app.button(key="freeze_qkv").click().run()
    assert not app.exception
    assert all(app.number_input(key=key).value == 0 for key in ("q_rank", "k_rank", "v_rank"))
    for page in ["测试 / benchmark", "历史测试记录", "任务进度"]:
        app.radio(key="page").set_value(page).run()
        assert not app.exception
    config = deepcopy(EVAL_DEFAULTS)
    config.update(
        model=dict(type="base", path=str(OUTPUT_ROOT / "smoke/tiny-base")),
        local_data=str(OUTPUT_ROOT / "smoke/data.json"),
        samples=1,
        max_new_tokens=6,
        num_beams=1,
        local_files_only=True,
        metrics=["BLEU"],
        smoke=True,
    )
    runs = launch([("tests", config), ("benchmarks", config)])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        statuses = [read_json(Path(r["path"]) / "status.json")["status"] for r in runs]
        if all(s in ("completed", "failed") for s in statuses):
            break
        time.sleep(0.5)
    assert statuses == ["completed", "completed"], statuses
    from utils.runs import history

    # 从现存冒烟运行选择 TensorBoard，避免归档后旧路径索引失效。
    run = next(
        record["path"]
        for record in history(include_smoke=True)
        if record["kind"] == "lora"
        and record["status"] == "completed"
        and record["config"].get("smoke")
        and list((Path(record["path"]) / "tensorboard").glob("events.*"))
    )
    url = tensorboard(run)
    deadline = time.monotonic() + 30
    success = False
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/data/plugin/scalars/tags", timeout=2) as response:
                tags = json.loads(response.read())
                assert tags
                success = True
                break
        except OSError:
            time.sleep(0.5)
    assert success, "TensorBoard did not expose training scalars"
    # 清理本次冒烟启动的 TensorBoard，保留此前已有的服务。
    import subprocess

    for server in OUTPUT_ROOT.rglob("tensorboard_server.json"):
        pid = read_json(server, {}).get("pid")
        if pid not in existing_pids and alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True)
    print(
        "PASS: four frontend pages, all-frozen control, "
        "detached sequential jobs, TensorBoard HTTP scalars"
    )


def training_pool_pipeline():
    """验证训练池参数快照、刷新保留、单项启动、整池串行训练与重复提交保护。"""
    import time
    from copy import deepcopy
    from uuid import uuid4

    from streamlit.testing.v1 import AppTest

    from config import OUTPUT_ROOT
    from utils import training_pool
    from utils.runs import write_json

    training_pool.POOL_ROOT = OUTPUT_ROOT / "smoke" / ("pool-" + uuid4().hex)
    app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
    app.text_input(key="pool_name").set_value("前端参数快照")
    app.button(key="enqueue_training").click().run()
    assert not app.exception
    entry = training_pool.entries()[0]
    assert entry["name"] == "前端参数快照" and entry["status"] == "pending"
    app.number_input(key="q_rank").set_value(8).run()
    assert training_pool.entries()[0]["config"]["ranks"]["q"] == 4
    config = deepcopy(TRAIN_DEFAULTS)
    config.update(
        model=str(OUTPUT_ROOT / "smoke/tiny-base"),
        local_data=str(OUTPUT_ROOT / "smoke/data.json"),
        max_steps=1,
        batch_size=2,
        eval_batch_size=2,
        logging_steps=1,
        warmup_steps=0,
        max_length=128,
        epochs=1,
        local_files_only=True,
        smoke=True,
    )
    entry["config"] = config
    write_json(training_pool.POOL_ROOT / f"{entry['id']}.json", entry)
    app.button(key="start_pool_" + entry["id"]).click().run()
    assert not app.exception

    def wait_finished(count):
        """等待微型后台训练结束，确认所有池项成功且按顺序执行。"""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            pool = training_pool.entries()
            if (
                len(pool) == count
                and all(item["status"] == "completed" for item in pool)
                and not (OUTPUT_ROOT / "jobs/active.lock").exists()
            ):
                return pool
            assert not any(item["status"] == "failed" for item in pool), pool
            time.sleep(0.5)
        raise AssertionError("训练池冒烟超时")

    wait_finished(1)
    first = training_pool.add("lora", config, "第二项 LoRA")
    second = training_pool.add("fft", config, "第三项 FFT")
    removable = training_pool.add("lora", config, "可移除项")
    training_pool.remove(removable["id"])
    config["learning_rate"] = 1e-8
    assert training_pool.entries()[1]["config"]["learning_rate"] != 1e-8
    app.button(key="refresh_training_pool").click().run()
    app.button(key="start_pool_all").click().run()
    assert not app.exception
    assert not app.error
    pool = wait_finished(3)
    statuses = [read_json(OUTPUT_ROOT / item["run"] / "status.json") for item in pool]
    assert statuses[1]["ended"] <= statuses[2]["started"]
    try:
        training_pool.start_pending([first["id"], second["id"]])
        raise AssertionError("已提交任务被重复启动")
    except ValueError:
        pass
    print("PASS: persistent training pool, snapshots, single/all UI launch, serial LoRA/FFT")


def resume_pipeline():
    """验证 LoRA / FFT 续训与连续训练结果一致，并检查前端续训按钮。"""
    from copy import deepcopy
    from unittest.mock import patch

    import torch
    from streamlit.testing.v1 import AppTest

    from config import OUTPUT_ROOT
    from utils.checkpoints import available
    from utils.models import load_model
    from utils.runs import history
    from utils.training import Progress, train

    def interrupt_saved(self, args, state, control, **kwargs):
        """完整保存首轮断点后模拟中断，保留原计划总步数。"""
        if state.global_step == 2:
            raise KeyboardInterrupt("smoke: interrupt after complete checkpoint")

    for kind in ("lora", "fft"):
        config = deepcopy(TRAIN_DEFAULTS)
        config.update(
            model=str(OUTPUT_ROOT / "smoke/tiny-base"),
            local_data=str(OUTPUT_ROOT / "smoke/data.json"),
            epochs=2,
            max_steps=-1,
            batch_size=2,
            eval_batch_size=2,
            warmup_steps=0,
            logging_steps=1,
            max_length=128,
            local_files_only=True,
            smoke=True,
            dropout=0,
            ranks=dict(q=2, k=0, v=1),
        )
        continuous = new_run(kind, config)
        train(config, continuous, kind)
        interrupted = new_run(kind, config)
        with patch.object(Progress, "on_save", interrupt_saved):
            try:
                train(config, interrupted, kind)
                raise AssertionError("未触发中断")
            except KeyboardInterrupt:
                pass
        assert read_json(interrupted / "status.json")["status"] == "interrupted"
        assert available(interrupted)[-1].name == "checkpoint-2"
        # 重现 PowerShell 七位小数历史状态，确保续训启动和累计耗时兼容 Python 3.10。
        status = read_json(interrupted / "status.json")
        status.pop("elapsed_seconds", None)
        status["started"] = "2026-10-12T14:25:54.4977937+08:00"
        status["ended"] = "2026-10-12T15:25:54.4977937+08:00"
        from utils.runs import write_json

        write_json(interrupted / "status.json", status)
        # 测试专用：显式允许冒烟记录，实际前端始终只显示正式运行。
        with patch("utils.runs.history", return_value=history(include_smoke=True)):
            app = AppTest.from_file(
                str(Path(__file__).with_name("app.py")), default_timeout=30
            ).run()
            app.radio(key="page").set_value("任务进度").run()
            index = next(
                i
                for i, item in enumerate(history(include_smoke=True))
                if item["path"] == str(interrupted)
            )
            app.selectbox(key="progress_run").set_value(index).run()
            app.button(key="resume_training_" + interrupted.name).click().run()
            assert not app.exception and not app.error
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = read_json(interrupted / "status.json")
            assert status["status"] != "failed", status
            if status["status"] == "completed" and not (OUTPUT_ROOT / "jobs/active.lock").exists():
                break
            time.sleep(0.5)
        assert status["status"] == "completed", status
        assert status["total_elapsed_seconds"] >= 3600
        rows = [
            json.loads(line) for line in (interrupted / "training.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in rows if "loss" in row] == [1, 2, 3, 4]
        assert read_json(interrupted / "attempts.json")[0]["status"]["status"] == "interrupted"
        expected, _ = load_model(dict(type=kind, path=str(continuous / "final")), True, "cpu")
        actual, _ = load_model(dict(type=kind, path=str(interrupted / "final")), True, "cpu")
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=1e-5, atol=1e-6)
    print("PASS: LoRA/FFT complete state resume matches continuous training; UI resume")


def stop_pipeline():
    """验证停止按钮保存 step 断点，并取消批次后续任务。"""
    from copy import deepcopy

    from streamlit.testing.v1 import AppTest

    from config import OUTPUT_ROOT
    from utils.checkpoints import available
    from utils.jobs import launch

    config = deepcopy(TRAIN_DEFAULTS)
    config.update(
        model=str(OUTPUT_ROOT / "smoke/tiny-base"),
        local_data=str(OUTPUT_ROOT / "smoke/data.json"),
        epochs=100,
        max_steps=2000,
        batch_size=2,
        eval_batch_size=2,
        warmup_steps=0,
        logging_steps=1,
        max_length=128,
        local_files_only=True,
        smoke=True,
    )
    runs = launch([("lora", config), ("fft", config)])
    first = Path(runs[0]["path"])
    deadline = time.monotonic() + 120
    while not (first / "progress.json").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (first / "progress.json").exists()
    app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
    app.radio(key="page").set_value("任务进度").run()
    app.button(key="stop_batch").click().run()
    assert not app.exception and not app.error
    deadline = time.monotonic() + 120
    while (OUTPUT_ROOT / "jobs/active.lock").exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not (OUTPUT_ROOT / "jobs/active.lock").exists()
    assert available(first)
    for run in runs:
        assert read_json(Path(run["path"]) / "status.json")["status"] == "interrupted"
    assert not (Path(runs[1]["path"]) / "training.jsonl").exists()
    print("PASS: UI stop, step checkpoint, child exit, later seed cancelled, lock released")


def formal_records_pipeline():
    """验证目录与标记双重排除冒烟，检查前端正式页面及 TensorBoard 聚合范围。"""
    from uuid import uuid4

    from streamlit.testing.v1 import AppTest

    from config import OUTPUT_ROOT
    from tensorboard_launcher import logdir_spec
    from utils.models import discover
    from utils.runs import history, write_json

    fixture = OUTPUT_ROOT / ("scan-fixture-" + uuid4().hex)
    fixture.mkdir()
    try:
        write_json(fixture / "config.json", dict(smoke=True))
        write_json(fixture / "status.json", dict(kind="fft", status="completed"))
        (fixture / "model.safetensors").touch()
        assert str(fixture) not in [model["path"] for model in discover()]
        assert str(fixture) not in [record["path"] for record in history()]
        assert str(fixture) in [model["path"] for model in discover(include_smoke=True)]
        assert str(fixture) in [record["path"] for record in history(include_smoke=True)]
        for model in discover()[1:]:
            assert "smoke" not in Path(model["path"]).relative_to(OUTPUT_ROOT).parts
            assert not model["training_config"].get("smoke")
        for record in history():
            assert "smoke" not in Path(record["path"]).relative_to(OUTPUT_ROOT).parts
            assert not record["config"].get("smoke")
        spec = logdir_spec()
        from tensorboard.backend.event_processing.data_ingester import _parse_event_files_spec

        for path in _parse_event_files_spec(spec):
            assert "smoke" not in Path(path).relative_to(OUTPUT_ROOT).parts
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
        for page in ("测试 / benchmark", "历史测试记录", "任务进度"):
            app.radio(key="page").set_value(page).run()
            assert not app.exception
        print("PASS: formal model/history filtering, legacy smoke flag, UI pages, TensorBoard spec")
    finally:
        for name in ("config.json", "status.json", "model.safetensors"):
            (fixture / name).unlink(missing_ok=True)
        fixture.rmdir()


def timestamps_pipeline():
    """验证七位小数、UTC 时区与坏时间记录不会阻断任务生命周期。"""
    from config import TRAIN_DEFAULTS
    from utils.runs import elapsed_between, write_json

    assert (
        elapsed_between(
            "2026-10-11T17:31:21.91346+08:00",
            "2026-10-12T15:25:54.4977937+08:00",
        )
        == 78872.584333
    )
    assert (
        elapsed_between("2026-10-12T14:25:54.4977937+08:00", "2026-10-12T15:25:54.4977937+08:00")
        == 3600
    )
    assert elapsed_between("2026-10-12T07:25:54.497793Z", "2026-10-12T15:25:55.497793+08:00") == 1
    assert elapsed_between("bad", "2026-10-12T15:25:54+08:00") is None
    assert elapsed_between("2026-10-12T16:00:00+08:00", "2026-10-12T15:00:00+08:00") is None
    config = {**TRAIN_DEFAULTS, "smoke": True}
    path = new_run("fft", config)
    write_json(path / "attempts.json", [dict(status=dict(started="bad", ended="bad"))])
    with running(path, "fft", config):
        print("Invalid historical time skipped without blocking task startup")
    state = read_json(path / "status.json")
    assert state["status"] == "completed" and state["timing_warnings"]
    print("PASS: timestamp precision/timezones, invalid history warning and task startup")


def medium_pipeline(kind):
    """用真实 GPT-2 Medium 执行指定模型类型的短训练或评测冒烟。"""
    from copy import deepcopy

    from config import EVAL_DEFAULTS, FFT_DEFAULTS, MODEL_NAME, OUTPUT_ROOT, TRAIN_DEFAULTS
    from utils.evaluation import evaluate
    from utils.runs import write_json
    from utils.training import train

    if kind == "base":
        spec = dict(type="base", path=MODEL_NAME, name="GPT-2 Medium real smoke Base")
    else:
        config = deepcopy(TRAIN_DEFAULTS if kind == "lora" else FFT_DEFAULTS)
        config.update(
            max_steps=1,
            train_samples=2,
            eval_samples=2,
            batch_size=1,
            eval_batch_size=1,
            gradient_accumulation=1,
            logging_steps=1,
            warmup_steps=0,
            max_length=128,
            precision="bf16",
            local_files_only=True,
            smoke=True,
        )
        path = new_run(kind, config)
        train(config, path, kind)
        spec = dict(
            type=kind,
            path=str(path / "final"),
            name=f"GPT-2 Medium real smoke {kind}",
            training_run=str(path),
        )
        write_json(OUTPUT_ROOT / f"smoke/medium-{kind}-run.json", dict(path=str(path)))
    config = deepcopy(EVAL_DEFAULTS)
    config.update(model=spec, samples=2, max_new_tokens=12, local_files_only=True, smoke=True)
    evaluation = new_run("benchmarks", config)
    evaluate(config, evaluation)
    assert read_json(evaluation / "status.json")["status"] == "completed"
    assert len(read_json(evaluation / "metrics.json")["metrics"]) == 8
    write_json(OUTPUT_ROOT / f"smoke/medium-{kind}-eval.json", dict(path=str(evaluation)))
    print(f"PASS: ACTUAL GPT-2 Medium {kind}, E2E samples, all metrics, telemetry and logs")


def guards_pipeline():
    """验证路径边界、全部冻结、非法参数、非有限梯度和评分失败的数据保留。"""
    import torch
    from transformers import GPT2LMHeadModel

    from config import OUTPUT_ROOT, PROJECT_ROOT, inside_project
    from utils.qkv_lora import attach, validate_ranks
    from utils.runs import estimate
    from utils.scoring import score
    from utils.training import Progress

    folder = OUTPUT_ROOT / "smoke/guards"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        inside_project(PROJECT_ROOT.parent / "gpt2-small-e2e-lora")
        raise AssertionError("Outside path accepted")
    except ValueError:
        pass
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_ROOT / "smoke/tiny-base").eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = model(ids).logits.detach()
    attach(model, dict(q=0, k=0, v=0))
    assert not any(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model(ids).logits, expected)
    try:
        validate_ranks(dict(q=-1, k=0, v=0), 1)
        raise AssertionError("Negative rank accepted")
    except ValueError:
        pass

    class State:
        """异常路径测试使用的最小 Trainer 状态对象，避免启动真实训练。"""

        global_step = 1
        epoch = 1

    try:
        Progress(folder).on_log(None, State(), None, logs={"grad_norm": float("nan")})
        raise AssertionError("NaN gradient accepted")
    except FloatingPointError:
        pass
    prediction = folder / "predictions.jsonl"
    prediction.write_text(
        json.dumps(dict(id=0, target="food", prediction="", references=["food"])) + "\n"
    )
    try:
        score(prediction, ["BLEU", "NIST"], folder)
        raise AssertionError("Undefined NIST should fail")
    except RuntimeError:
        pass
    result = read_json(folder / "metrics.json")
    assert "BLEU-4" in result["metrics"] and "NIST" in result["errors"]
    assert prediction.exists() and (folder / "sample_scores.jsonl").exists()
    assert estimate(TRAIN_DEFAULTS, "lora") == (None, 0)
    print(
        "PASS: project containment, all-frozen logits, invalid ranks, "
        "NaN rejection, partial-score retention, smoke excluded from ETA"
    )


if __name__ == "__main__":
    require_lora()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "runtime",
            "models",
            "lora",
            "fft",
            "evaluation",
            "frontend",
            "training-pool",
            "resume",
            "stop",
            "formal-records",
            "timestamps",
            "medium-base",
            "medium-lora",
            "medium-fft",
            "guards",
        ],
    )
    args = parser.parse_args()
    {
        "runtime": runtime,
        "models": model_pipeline,
        "lora": lambda: training_pipeline("lora"),
        "fft": lambda: training_pipeline("fft"),
        "evaluation": evaluation_pipeline,
        "frontend": frontend_pipeline,
        "training-pool": training_pool_pipeline,
        "resume": resume_pipeline,
        "stop": stop_pipeline,
        "formal-records": formal_records_pipeline,
        "timestamps": timestamps_pipeline,
        "medium-base": lambda: medium_pipeline("base"),
        "medium-lora": lambda: medium_pipeline("lora"),
        "medium-fft": lambda: medium_pipeline("fft"),
        "guards": guards_pipeline,
    }[args.stage]()
