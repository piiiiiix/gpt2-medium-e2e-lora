"""断点检查模块：只选择同一训练运行内包含完整恢复状态的 checkpoint。"""

from config import inside_project
from utils.runs import read_json


def available(run):
    """按训练步数列出完整断点，排除未写完或缺少优化器状态的目录。"""
    run = inside_project(run)
    config = read_json(run / "config.json", {})
    weight = (
        "adapter_model.safetensors"
        if config.get("ranks") and (read_json(run / "status.json", {}).get("kind") == "lora")
        else "model.safetensors"
    )
    result = []
    for path in (run / "checkpoints").glob("checkpoint-*"):
        if not path.name.removeprefix("checkpoint-").isdigit():
            continue
        required = [weight, "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]
        if not all((path / name).is_file() for name in required):
            continue
        state = read_json(path / "trainer_state.json", {})
        if state.get("global_step") != int(path.name.split("-")[-1]):
            continue
        result.append(path)
    return sorted(result, key=lambda path: int(path.name.split("-")[-1]))


def validate(run, checkpoint):
    """拒绝项目外、其他运行或不完整断点，返回可用于完整续训的路径。"""
    checkpoint = inside_project(checkpoint)
    if checkpoint not in available(run):
        raise ValueError("请选择此运行内完整的 checkpoint（含优化器、调度器与随机状态）。")
    return checkpoint
