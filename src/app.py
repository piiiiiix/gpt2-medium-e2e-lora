"""Streamlit 单文件前端：提供训练、评测及统一任务列表页面。

前端只组织参数与展示结果，实际任务交给独立后端进程执行。"""

import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st

from config import (
    EVAL_DEFAULTS,
    FFT_DEFAULTS,
    METRICS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    TRAIN_DEFAULTS,
    require_lora,
)
from utils import training_pool
from utils.checkpoints import available
from utils.jobs import alive, launch, resume_training, stop_batch, tensorboard
from utils.models import discover
from utils.runs import HK, elapsed_between, estimate, history, read_json
from utils.task_list import filter_tasks, task_status, time_label

require_lora()
st.set_page_config(page_title="GPT-2 Medium · E2E", layout="wide")
st.title("GPT-2 Medium · E2E")
st.caption("训练、评测与实验记录 · conda lora")
page = st.sidebar.radio("菜单", ["训练", "测试 / benchmark", "任务列表"], key="page")
st.sidebar.caption(str(PROJECT_ROOT))


@st.cache_data(ttl=5, max_entries=2)
def models(include_checkpoints=False):
    """缓存项目内的模型列表，避免前端每次交互都重复扫描磁盘。"""
    return discover(include_checkpoints=include_checkpoints)


@st.cache_data(ttl=2, max_entries=2)
def records():
    """短时缓存运行历史，兼顾页面刷新速度与状态更新。"""
    return history()


def start(tasks):
    """将前端选择的任务交给后台顺序执行，并显示启动结果。"""
    try:
        runs = launch(tasks)
        st.session_state["last_runs"] = runs
        records.clear()
        st.success(f"已启动 {len(runs)} 个任务，将按顺序执行。请到“任务列表”查看。")
    except Exception as exc:
        st.error(str(exc))


def tb_button(path):
    """按训练运行准备 TensorBoard，只在用户点击时启动服务。"""
    if not list((path / "tensorboard").glob("events.*")):
        st.caption("暂无 TensorBoard 日志；任务开始并产生日志后可查看。")
        return
    key = "tb_" + str(path)
    if st.button("打开 TensorBoard", key=key):
        try:
            with st.spinner("正在准备训练 TensorBoard…"):
                st.session_state[key + "_url"] = tensorboard(path)
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get(key + "_url"):
        st.link_button("进入 TensorBoard", st.session_state[key + "_url"])


def duration(config, kind):
    """根据同配置成功记录的均值，展示任务耗时与预计完成时间。"""
    if st.button("按历史均值估算耗时", key="estimate_" + kind):
        seconds, count = estimate(config, kind)
        if seconds is None:
            st.info("暂无同配置、同硬件历史估算。")
        else:
            eta = datetime.now(HK) + timedelta(seconds=seconds)
            st.info(
                f"历史均值 {seconds / 60:.1f} 分钟（{count} 次）；"
                f"预计完成 {eta:%Y-%m-%d %H:%M:%S}。"
            )


def task_details(record):
    """展示单条任务的结果、下载、续训、进度、日志和 GPU 采样。"""
    path = Path(record["path"])
    if (path / "model_removal.json").exists():
        st.warning("此运行的模型及断点已清理；配置和日志仅保留用于追溯。")
    st.json({k: v for k, v in record.items() if k != "config"})
    if record["config"].get("smoke"):
        st.caption("冒烟测试记录")
    st.json(read_json(path / "metrics.json", {}))
    with st.expander("运行配置"):
        st.json(record["config"])
    for filename in [
        "predictions.jsonl",
        "sample_scores.jsonl",
        "metrics.json",
        "metrics.csv",
        "config.json",
        "run.log",
        "gpu.jsonl",
    ]:
        file = path / filename
        if file.exists():
            st.download_button(
                "下载 " + filename,
                file.read_bytes(),
                file_name=filename,
                key="download_" + str(path) + filename,
            )
    scores = path / "sample_scores.jsonl"
    if scores.exists():
        frame = pd.read_json(scores, lines=True)
        st.dataframe(frame, alt="逐样本预测、参考答案及测试成绩")
    if (path / "predictions.jsonl").exists():
        st.caption("重新评分只计算已有预测，不补生成。预测不完整时请重新启动完整评测。")
        rescore = st.multiselect(
            "重新评分指标",
            METRICS,
            default=record["config"].get("metrics", METRICS),
            key="rescore_" + str(path),
        )
        if st.button("使用已保存预测重新评分", key="score_" + str(path)):
            config = {
                **record["config"],
                "predictions": str(path / "predictions.jsonl"),
                "metrics": rescore,
            }
            if rescore:
                start([("benchmarks", config)])
            else:
                st.error("至少选择一个指标。")

    if record["kind"] in ("lora", "fft"):
        tb_button(path)
        st.json(read_json(path / "data_summary.json", {}))
        st.json(read_json(path / "result.json", {}))
    stopped = record["status"] in ("interrupted", "failed") or (
        record["status"] == "running" and not alive(record.get("pid"))
    )
    if record["kind"] in ("lora", "fft") and stopped:
        checkpoints = available(path)
        if checkpoints:
            checkpoint = st.selectbox(
                "续训 checkpoint",
                checkpoints,
                index=len(checkpoints) - 1,
                format_func=lambda p: p.name,
                key="resume_checkpoint_" + path.name,
            )
            st.caption("恢复原参数、权重、优化器、学习率调度器与步数；不会自动继续其他 seed。")
            if st.button("断点续训", key="resume_training_" + path.name):
                try:
                    resume_training(path, checkpoint)
                    records.clear()
                    st.success("已提交断点续训，请刷新任务状态。")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("没有完整 checkpoint，无法断点续训；未开始的 seed 请重新加入待训练池。")
    if record["status"] == "running" and not alive(record.get("pid")):
        st.warning("任务进程已退出，未写入完成状态，请查看 process.log。")
    state = read_json(path / "progress.json", {})
    if state.get("total"):
        st.progress(min(state["completed"] / state["total"], 1.0), text=state.get("phase", ""))
    seconds = state.get("remaining_seconds") if record["status"] == "running" else None
    if seconds is None and record["status"] in ("queued", "running"):
        average, count = estimate(record["config"], record["kind"])
        if average is not None:
            elapsed = (
                (elapsed_between(record["started"], datetime.now(HK).isoformat()) or 0)
                if record.get("started")
                else 0
            )
            seconds = max(0, average - elapsed)
            st.caption(f"依据 {count} 次同配置成功运行的耗时均值")
    if seconds is not None:
        eta = datetime.now(HK) + timedelta(seconds=seconds)
        st.metric("预计完成时间", eta.strftime("%H:%M:%S"))
    elif record["status"] in ("queued", "running"):
        st.caption("暂无历史估算，等待运行进度。")
    if record.get("elapsed_seconds") is not None:
        elapsed = record.get("total_elapsed_seconds", record["elapsed_seconds"])
        st.metric("累计运行耗时（秒）", f"{elapsed:.1f}")
    for filename in ["run.log", "process.log"]:
        if (path / filename).exists():
            st.caption("训练内部日志" if filename == "run.log" else "后台进程日志（含重复输出）")
            with (path / filename).open("rb") as stream:
                stream.seek(max(0, (path / filename).stat().st_size - 16000))
                content = stream.read().decode("utf-8", errors="replace")
            st.code(content, language="text")
    if (path / "gpu.jsonl").exists():
        try:
            gpu = pd.read_json(path / "gpu.jsonl", lines=True)
            st.dataframe(gpu.tail(20), alt="GPU 占用率、核心温度及显存采样")
            for label, column in [
                ("GPU 占用率（%）", "utilization_percent"),
                ("核心温度（°C）", "temperature_celsius"),
                ("显存（MiB）", "memory_used_mib"),
            ]:
                if column in gpu and "gpu" in gpu:
                    chart = gpu[["time", "gpu", column]].copy()
                    chart[column] = pd.to_numeric(chart[column], errors="coerce")
                    st.line_chart(chart, x="time", y=column, color="gpu", alt=label + "随时间变化")
            if "wddm_memory" in gpu:
                rows, reasons = [], set()
                for _, sample in gpu.iterrows():
                    memory = sample.get("wddm_memory")
                    if not isinstance(memory, dict):
                        continue
                    if memory.get("unavailable_reason"):
                        reasons.add(memory["unavailable_reason"])
                    for adapter in memory.get("adapters", []):
                        rows.append({"time": sample["time"], "pid": memory["pid"], **adapter})
                st.caption("WDDM 按适配器 LUID 区分；进程值仅包含本任务 PID，缺失不代表零。")
                if rows:
                    memory_frame = pd.DataFrame(rows)
                    st.dataframe(memory_frame.tail(20), alt="WDDM 专用与共享 GPU 内存")
                    for column, label in [
                        ("adapter_shared_mib", "整卡共享 GPU 内存（MiB）"),
                        ("process_shared_mib", "本任务共享 GPU 内存（MiB）"),
                        ("adapter_dedicated_mib", "整卡专用 GPU 内存（MiB）"),
                        ("process_dedicated_mib", "本任务专用 GPU 内存（MiB）"),
                    ]:
                        if column in memory_frame:
                            st.caption(label)
                            st.line_chart(
                                memory_frame, x="time", y=column, color="adapter", alt=label
                            )
                for reason in sorted(reasons):
                    st.caption(f"WDDM 内存采样不可用：{reason}")
            else:
                st.caption("此历史记录未采集共享 GPU 内存。")
        except ValueError:
            st.caption("GPU 采样正在写入。")


if page == "训练":
    kind = (
        st.segmented_control("训练方式", ["lora", "fft"], default="lora", key="train_kind")
        or "lora"
    )
    config = deepcopy(TRAIN_DEFAULTS if kind == "lora" else FFT_DEFAULTS)
    profile = st.selectbox("训练配置", ["论文默认 / Prefix-Tuning FT 默认", "8GB 显存适配"])
    if profile == "8GB 显存适配":
        config.update(
            batch_size=1,
            gradient_accumulation=8 if kind == "lora" else 10,
            eval_batch_size=1,
            precision="bf16",
            gradient_checkpointing=True,
        )
    st.caption("LoRA 默认值来自 V2 表 11；FFT 默认值参考 Prefix-Tuning 的 E2E FT 及作者代码。")
    if kind == "fft":
        st.caption(
            "FT：保留作者原始文本空白、动态 padding；每 5000 步验证，"
            "每 500000 步保存断点，final 使用训练结束权重（不重载最佳 epoch）。"
        )
    if kind == "lora":
        st.subheader("Q、K、V 的 LoRA rank")
        st.caption("三个设置统一应用到所有 Transformer 层。rank=0：冻结该投影，不挂 LoRA。")

        def freeze_qkv():
            """前端回调：将 Q、K、V 三个 rank 同时设为 0。"""
            for key in ("q_rank", "k_rank", "v_rank"):
                st.session_state[key] = 0

        for key, default in [("q_rank", 4), ("k_rank", 0), ("v_rank", 4)]:
            st.session_state.setdefault(key, default)
        with st.container(horizontal=True):
            q = st.number_input("Q rank", 0, 1024, value=None, step=1, key="q_rank")
            k = st.number_input("K rank", 0, 1024, value=None, step=1, key="k_rank")
            v = st.number_input("V rank", 0, 1024, value=None, step=1, key="v_rank")
            st.button("全部冻结", key="freeze_qkv", on_click=freeze_qkv)
        config["ranks"] = dict(q=int(q or 0), k=int(k or 0), v=int(v or 0))
    with st.form(f"training_config_{kind}_{profile}"):
        config["dataset"] = st.selectbox(
            "训练数据", ["lora/e2e-original", "GEM/e2e_nlg"], key=f"train_dataset_{kind}"
        )
        repeat_seeds = st.text_input(
            "重复实验 seeds（逗号分隔；留空使用下方 Seed）",
            "",
            help="论文使用3个种子取均值；具体种子组合未公开，例如可用110,111,112。",
            key=f"repeat_seeds_{kind}_{profile}",
        )
        a, b, c = st.columns(3)
        # 加减按钮每次调整 1e-6；更小的学习率仍可直接输入。
        config["learning_rate"] = a.number_input(
            "学习率",
            min_value=1e-9,
            value=config["learning_rate"],
            step=1e-6,
            format="%.9f",
            key=f"learning_rate_{kind}_{profile}",
            help="加减步长为 0.000001，可直接输入更小的值。",
        )
        config["epochs"] = b.number_input("Epochs", min_value=0.1, value=float(config["epochs"]))
        config["seed"] = c.number_input("Seed", min_value=0, value=config["seed"])
        config["batch_size"] = a.number_input(
            "每设备 batch size", min_value=1, value=config["batch_size"]
        )
        config["gradient_accumulation"] = b.number_input(
            "梯度累积", min_value=1, value=config["gradient_accumulation"]
        )
        config["eval_batch_size"] = c.number_input(
            "验证 batch size", min_value=1, value=config["eval_batch_size"]
        )
        config["warmup_steps"] = a.number_input(
            "Warmup steps", min_value=0, value=config["warmup_steps"]
        )
        config["weight_decay"] = b.number_input(
            "Weight decay", min_value=0.0, value=config["weight_decay"]
        )
        config["label_smoothing"] = c.number_input(
            "Label smoothing", min_value=0.0, max_value=0.99, value=config["label_smoothing"]
        )
        config["dropout"] = a.number_input(
            "LoRA dropout" if kind == "lora" else "GPT-2 dropout",
            min_value=0.0,
            max_value=0.99,
            value=config["dropout"],
        )
        config["max_grad_norm"] = b.number_input(
            "梯度裁剪（0=关闭）",
            min_value=0.0,
            value=config["max_grad_norm"],
            key=f"max_grad_norm_{kind}_{profile}",
        )
        config["adam_epsilon"] = c.number_input(
            "Adam epsilon",
            min_value=1e-12,
            value=config["adam_epsilon"],
            format="%.10f",
            key=f"adam_epsilon_{kind}_{profile}",
        )
        with st.expander("输入协议与优化器"):
            protocols = ["lora_e2e" if kind == "lora" else "prefix_ft", "legacy"]
            config["protocol"] = st.selectbox(
                "训练输入协议",
                protocols,
                index=protocols.index(config["protocol"]),
                key=f"protocol_{kind}_{profile}",
            )
            if kind == "fft":
                config["source_ft"] = config["protocol"] == "prefix_ft"
            config["optimizer"] = st.selectbox(
                "AdamW 实现", ["source_adamw", "adamw_torch"], key=f"optimizer_{kind}_{profile}"
            )
            for beta in ("adam_beta1", "adam_beta2"):
                config[beta] = st.number_input(
                    beta,
                    min_value=0.0,
                    max_value=0.99999,
                    value=config[beta],
                    key=f"{beta}_{kind}_{profile}",
                )
        if kind == "lora":
            config["alpha"] = b.number_input("LoRA alpha", min_value=1, value=config["alpha"])
        config["precision"] = c.selectbox(
            "精度",
            ["fp32", "fp16", "bf16"],
            index=["fp32", "fp16", "bf16"].index(config["precision"]),
        )
        config["gradient_checkpointing"] = st.checkbox(
            "梯度检查点", value=config["gradient_checkpointing"]
        )
        config["max_length"] = st.number_input(
            "最大训练长度", min_value=32, max_value=1024, value=config["max_length"]
        )
        config["logging_steps"] = st.number_input(
            "日志间隔（steps）", min_value=1, value=config["logging_steps"]
        )
        with st.expander("少量样本 / 冒烟设置"):
            n = st.number_input("训练样本数，0=全部", min_value=0, value=0)
            e = st.number_input("验证样本数，0=全部", min_value=0, value=0)
            steps = st.number_input("最大 steps，0=按 epochs", min_value=0, value=0)
            config.update(train_samples=n or None, eval_samples=e or None, max_steps=steps or -1)
            config["smoke"] = st.checkbox("将本次运行标记为冒烟测试", value=False)
        pool_name = st.text_input("任务名称（可选）", key="pool_name")
        submitted = st.form_submit_button("启动训练", type="primary")
        enqueue = st.form_submit_button("加入待训练池", key="enqueue_training")
    st.caption(f"有效 batch size：{config['batch_size'] * config['gradient_accumulation']}。")
    duration(config, kind)
    if submitted or enqueue:
        try:
            seeds = (
                [int(s.strip()) for s in repeat_seeds.split(",")]
                if repeat_seeds.strip()
                else [config["seed"]]
            )
            if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
                raise ValueError("seeds 必须为互不重复的非负整数")
        except ValueError as exc:
            st.error(f"Seed 输入错误：{exc}")
            st.stop()
        if kind == "lora" and not any(config["ranks"].values()):
            st.error("全部投影已冻结，没有可训练参数；请启用至少一个 rank，或使用 Base 测试。")
        elif submitted:
            start([(kind, {**deepcopy(config), "seed": seed}) for seed in seeds])
        else:
            try:
                for seed in seeds:
                    training_pool.add(
                        kind,
                        {**deepcopy(config), "seed": seed},
                        f"{pool_name} seed={seed}" if len(seeds) > 1 else pool_name,
                    )
                st.success("已保存参数快照到待训练池。")
            except Exception as exc:
                st.error(str(exc))

    st.subheader("待训练池")
    st.button("刷新训练池状态", key="refresh_training_pool")
    st.caption("参数保存到 medium 的 outputs；全部开始按添加顺序串行执行，关闭页面不影响训练。")
    pool = training_pool.entries()
    pending = [entry for entry in pool if entry["status"] == "pending"]
    if st.button("全部开始", key="start_pool_all", disabled=not pending):
        try:
            st.session_state["last_runs"] = training_pool.start_pending(
                [entry["id"] for entry in pending]
            )
            records.clear()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if not pool:
        st.info("待训练池为空，先配置参数并点击“加入待训练池”。")
    for entry in pool:
        with st.expander(f"{entry['name']} · {entry['kind']} · {entry['status']}"):
            st.json(entry["config"])
            if entry.get("run"):
                st.caption(f"运行目录：{OUTPUT_ROOT / entry['run']}")
            if entry["status"] == "pending":
                if st.button("开始这一项", key="start_pool_" + entry["id"]):
                    try:
                        st.session_state["last_runs"] = training_pool.start_pending([entry["id"]])
                        records.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if st.button("移除", key="remove_pool_" + entry["id"]):
                    training_pool.remove(entry["id"])
                    st.rerun()

elif page == "测试 / benchmark":
    if st.button("刷新模型列表"):
        models.clear()
    include_checkpoints = st.checkbox("显示训练中间 checkpoint", value=False)
    options = models(include_checkpoints)
    st.caption("默认仅列出最终模型和 Base；每个 checkpoint 是同一次训练的中间权重。")
    selected = st.multiselect(
        "选择模型（仅 medium 项目）",
        range(len(options)),
        default=[0],
        format_func=lambda i: f"{options[i]['type']} · {options[i]['name']}",
    )
    config = deepcopy(EVAL_DEFAULTS)
    from utils.protocols import GENERATION

    config["generation_profile"] = st.selectbox(
        "生成协议",
        ["auto", *GENERATION],
        help="auto 跟随模型训练协议；无协议标记的历史模型使用 legacy。",
    )
    manual_generation = st.checkbox("覆盖协议的生成参数", value=False)
    preset = GENERATION.get(config["generation_profile"], GENERATION["lora_paper"])
    st.caption("LoRA 论文档：beam=10、长度惩罚=0.9；Prefix FT：beam=5、长度惩罚=1.0。")
    with st.form("evaluation_config"):
        config["dataset"] = st.selectbox("评测数据", ["lora/e2e-original", "GEM/e2e_nlg"])
        config["scoring_profile"] = st.selectbox(
            "评分协议",
            ["e2e_official", "legacy_single"],
            help="官方：全部参考、MTEval、PTB与Java METEOR；legacy仅用于历史对照。",
        )
        kind = st.selectbox("任务", ["benchmarks", "tests"])
        config["split"] = st.selectbox("数据 split", ["test", "validation"])
        config["metrics"] = st.multiselect("评测指标", METRICS, default=METRICS)
        config["samples"] = st.number_input("样本数，0=全部", min_value=0, value=0) or None
        if manual_generation:
            for key, label, minimum in [
                ("max_new_tokens", "最大新 token 数", 1),
                ("num_beams", "Beam size", 1),
                ("length_penalty", "Length penalty", 0.0),
                ("no_repeat_ngram_size", "No repeat ngram size", 0),
            ]:
                config[key] = st.number_input(
                    label,
                    min_value=minimum,
                    value=preset[key],
                    key=f"eval_{key}_{config['generation_profile']}",
                )
        config["smoke"] = st.checkbox("将本次评测标记为冒烟测试", value=False)
        submitted = st.form_submit_button("启动评测", type="primary")
    if selected:
        duration({**config, "model": options[selected[0]]}, kind)
    if submitted:
        if not selected or not config["metrics"]:
            st.error("请选择模型及至少一个指标。")
        else:
            start([(kind, {**deepcopy(config), "model": options[i]}) for i in selected])

else:
    st.subheader("任务列表")
    st.caption("按创建时间倒序；关闭页面不影响后台任务。中断及异常退出归入失败。")
    if st.button("停止当前批次", key="stop_batch", type="primary"):
        try:
            stop_batch()
            st.success("停止请求已发送，后续排队项不会启动。")
        except Exception as exc:
            st.error(str(exc))
    st.caption("停止会取消整个批次；训练优先在 step 结束后保存断点，90 秒超时则强制停止。")

    @st.fragment(run_every=3)
    def task_list():
        """定时刷新任务列表，组合筛选并独立展开各条任务。"""
        if st.button("重置筛选", key="reset_tasks"):
            for key in ("task_kind", "task_model", "task_status"):
                st.session_state[key] = "全部"
        for key in ("task_kind", "task_model", "task_status"):
            st.session_state.setdefault(key, "全部")
        kind = (
            st.segmented_control(
                "任务类型", ["全部", "训练", "benchmark"], default=None, key="task_kind"
            )
            or "全部"
        )
        model = (
            st.segmented_control(
                "模型类型", ["全部", "base", "FFT", "LoRA"], default=None, key="task_model"
            )
            or "全部"
        )
        status = (
            st.segmented_control(
                "任务状态",
                ["全部", "已完成", "运行中", "队列中", "失败"],
                default=None,
                key="task_status",
            )
            or "全部"
        )
        if kind == "训练" and model == "base":
            st.info("基础模型无训练任务，请切换模型类型。")
        items = filter_tasks(history(), kind, model, status)
        st.caption(f"共 {len(items)} 条任务 · 创建时间倒序")
        if not items:
            st.info("暂无符合条件的任务。")
        for record in items:
            path = Path(record["path"])
            training = record["kind"] in ("lora", "fft")
            model_type = (
                record["kind"]
                if training
                else record["config"].get("model", {}).get("type", "未知")
            )
            label = (
                f"{path.name} · {'训练' if training else 'benchmark'} · {model_type} · "
                f"{task_status(record)} · {time_label(record)}"
            )
            panel = st.expander(label, key="task_" + str(path), on_change="rerun")
            if panel.open:
                with panel:
                    task_details(record)

    task_list()
