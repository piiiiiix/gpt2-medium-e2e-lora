"""项目配置中心：定义模型、数据、训练和评测默认值以及路径边界。

LoRA 论文参数与实现补充值的来源见项目根目录的参数来源.md。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_NAME = "openai-community/gpt2-medium"
DATASET_NAME = "lora/e2e-original"
METRICS = ["BLEU", "NIST", "METEOR", "ROUGE-L", "CIDEr"]
TRAIN_DEFAULTS = dict(
    model=MODEL_NAME,
    dataset=DATASET_NAME,
    learning_rate=2e-4,  # noqa: C408
    epochs=5,
    batch_size=8,
    eval_batch_size=4,
    gradient_accumulation=1,
    weight_decay=0.01,
    warmup_steps=500,
    protocol="lora_e2e",
    source_training=True,
    optimizer="source_adamw",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    label_smoothing=0.1,
    dropout=0.1,
    alpha=32,
    seed=110,
    precision="fp32",
    max_length=512,
    gradient_checkpointing=False,
    logging_steps=100,
    max_steps=-1,
    smoke=False,
    train_samples=None,
    eval_samples=None,
    local_data=None,
    local_files_only=False,
    ranks=dict(q=4, k=0, v=4),
)  # noqa: C408
# FFT 对齐 Li & Liang (2021) E2E FT；未在论文报告的值按作者代码补充。
# warmup=0 已核对作者仓库 training_args.py 与普通 E2E 启动路径。
FFT_DEFAULTS = {
    **TRAIN_DEFAULTS,
    "learning_rate": 5e-5,
    "batch_size": 10,
    "eval_batch_size": 10,
    "weight_decay": 0.0,
    "warmup_steps": 0,
    "label_smoothing": 0.0,
    "seed": 101,
    "max_length": 1024,
    "gradient_checkpointing": False,
    "protocol": "prefix_ft",
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
}
EVAL_DEFAULTS = dict(
    dataset=DATASET_NAME,
    split="test",
    scoring_profile="e2e_official",
    samples=None,  # noqa: C408
    metrics=METRICS.copy(),
    generation_profile="auto",
    max_new_tokens=None,
    num_beams=None,
    length_penalty=None,
    no_repeat_ngram_size=None,
    seed=42,
    local_data=None,
    local_files_only=False,
    model=dict(type="base", path=MODEL_NAME, name="GPT-2 Medium"),
    predictions=None,
    smoke=False,
)  # noqa: C408
GPU_INTERVAL_SECONDS = 2.0


def require_lora():
    """检查当前解释器是否属于 conda lora 环境，防止误用其他 Python。"""
    import sys

    if Path(sys.prefix).name.lower() != "lora":
        raise RuntimeError(f"Use conda environment lora, current interpreter: {sys.executable}")


def inside_project(path):
    """解析并校验路径，拒绝越出 medium 项目根目录的文件访问。"""
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path outside medium project: {path}")
    return path
