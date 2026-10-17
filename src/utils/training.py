"""训练后端：共享 LoRA / FFT 数据流程与 Trainer，保存模型、日志和 TensorBoard。"""

import json
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import TensorBoardCallback

from utils.data import load_data, subset, tokenize
from utils.protocols import save_protocol
from utils.qkv_lora import attach, restore, save_adapter
from utils.runs import running, write_json


class Progress(TrainerCallback):
    """Trainer 回调：保存逐步训练日志和进度，并检测非有限损失或梯度。"""

    def __init__(self, path):
        """设置本次运行路径与进度计时起点。"""
        self.path = path
        self.started = time.perf_counter()
        self.initial_step = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """保存 Trainer 日志，并在 loss、eval_loss 或 grad_norm 非有限时终止任务。"""
        with (self.path / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"step": state.global_step, "epoch": state.epoch, **(logs or {})}, default=str
                )
                + "\n"
            )
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in (logs or {}) and not math.isfinite(float(logs[key])):
                raise FloatingPointError(
                    f"Non-finite {key}; inspect training.jsonl. "
                    f"Try bf16/fp32 or adjust configuration."
                )

    def on_step_end(self, args, state, control, **kwargs):
        """每个训练 step 结束后保存进度，并按已用时间估算剩余时间。"""
        elapsed = time.perf_counter() - self.started
        if (self.path / "stop.request.json").exists():
            control.should_training_stop = True
            control.should_save = True
        if self.initial_step is None:
            self.initial_step = state.global_step - 1
        write_json(
            self.path / "progress.json",
            dict(
                completed=state.global_step,
                total=state.max_steps,
                phase="training",
                remaining_seconds=elapsed
                / max(1, state.global_step - self.initial_step)
                * (state.max_steps - state.global_step),
            ),
        )


class SourceTrainer(Trainer):
    """显式选择作者版 AdamW，历史配置继续使用 torch AdamW。"""

    def __init__(self, *args, run_config=None, **kwargs):
        self.run_config = run_config or {}
        super().__init__(*args, **kwargs)
        if self.run_config.get("source_training") and self.run_config.get("protocol") == "lora_e2e":
            self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        if self.run_config.get("optimizer") != "source_adamw":
            return super().create_optimizer()
        if self.optimizer is None:
            from utils.source_optimizer import SourceAdamW

            # LoRA 只训练 A/B；FFT 按作者 Trainer 排除 bias 与 LayerNorm。
            decay_names = self.get_decay_parameter_names(self.model)
            groups = [
                dict(
                    params=[
                        p
                        for n, p in self.model.named_parameters()
                        if p.requires_grad and (n in decay_names) == decay
                    ],
                    weight_decay=self.args.weight_decay if decay else 0.0,
                )
                for decay in (True, False)
            ]
            self.optimizer = SourceAdamW(
                groups,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.run_config.get("source_training") and self.run_config.get("protocol") == "lora_e2e":
            from utils.source_training import source_loss

            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            smoothing = self.run_config["label_smoothing"] if model.training else 0.0
            loss = source_loss(outputs.logits, inputs["labels"], smoothing)
            return (loss, outputs) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _get_train_sampler(self, train_dataset=None):
        if self.run_config.get("source_training") and self.run_config.get("protocol") == "lora_e2e":
            from utils.source_training import SourceSampler

            return SourceSampler(
                train_dataset if train_dataset is not None else self.train_dataset,
                num_replicas=1,
                rank=0,
                seed=self.args.seed,
            )
        return super()._get_train_sampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        if self.run_config.get("source_training") and self.run_config.get("protocol") == "lora_e2e":
            from torch.utils.data.distributed import DistributedSampler

            return DistributedSampler(eval_dataset, num_replicas=1, rank=0, seed=self.args.seed)
        return super()._get_eval_sampler(eval_dataset)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        save_protocol(output_dir or self.args.output_dir, self.run_config)


class AdapterTrainer(SourceTrainer):
    """LoRA 专用 Trainer：只保存 adapter，按验证结果重载最佳 adapter checkpoint。"""

    def __init__(self, *args, run_config=None, **kwargs):
        """保留 adapter 配置，初始化通用 Trainer。"""
        self.run_config = run_config
        super().__init__(*args, run_config=run_config, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        """保存当前 adapter 与 tokenizer，避免将修改后的合并层当作普通整模型保存。"""
        save_adapter(self.model, output_dir or self.args.output_dir, self.run_config)
        save_protocol(output_dir or self.args.output_dir, self.run_config)
        if self.processing_class:
            self.processing_class.save_pretrained(output_dir or self.args.output_dir)

    def _load_best_model(self):
        """训练结束后恢复验证损失最低的 adapter checkpoint。"""
        if self.state.best_model_checkpoint:
            restore(self.model, self.state.best_model_checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """恢复自定义 QKV adapter；优化器、调度器和随机状态由 Trainer 恢复。"""
        restore(model if model is not None else self.model, resume_from_checkpoint)


def validate(config, kind):
    """训练前检查 batch、精度、label smoothing 和 QKV 配置，拒绝无可训练参数。"""
    for key in [
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "logging_steps",
        "max_length",
    ]:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["epochs"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32/fp16/bf16")
    if config["precision"] != "fp32" and not torch.cuda.is_available():
        raise ValueError("Select fp32 on CPU")
    if not 0 <= config["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0,1)")
    if config.get("protocol", "legacy") not in (
        "legacy",
        "lora_e2e" if kind == "lora" else "prefix_ft",
    ):
        raise ValueError("Training protocol must match lora/fft, or use legacy")
    if config.get("optimizer", "adamw_torch") not in ("source_adamw", "adamw_torch"):
        raise ValueError("Unknown optimizer")
    for key, default in (("adam_epsilon", 1e-8), ("max_grad_norm", 1.0)):
        value = config.get(key, default)
        if not math.isfinite(value) or value < 0 or (key == "adam_epsilon" and value == 0):
            raise ValueError(f"Invalid {key}")
    for key in ("adam_beta1", "adam_beta2"):
        if not 0 <= config.get(key, 0.9) < 1:
            raise ValueError(f"Invalid {key}")
    if kind == "lora":
        if not isinstance(config["ranks"], dict):
            raise ValueError(
                'Use one model-wide ranks object: {"q":4,"k":0,"v":4}; '
                "per-layer settings are not supported for new training."
            )
        from utils.qkv_lora import validate_ranks

        validate_ranks(config["ranks"], 1)
        if not any(v > 0 for v in config["ranks"].values()):
            raise ValueError(
                "All QKV projections are frozen. Use Base "
                "testing, or enable at least one rank for training."
            )


def train(config, path, kind):
    """运行完整训练链路：数据处理、模型构建、训练验证、最佳模型保存和日志落盘。"""
    checkpoint = config.get("resume_from_checkpoint")
    if checkpoint:
        from utils.checkpoints import validate as validate_checkpoint

        checkpoint = str(validate_checkpoint(path, checkpoint))
    with running(path, kind, config):
        validate(config, kind)
        set_seed(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], local_files_only=config["local_files_only"]
        )
        if config.get("protocol") == "prefix_ft":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
        raw = load_data(config)
        if config["dataset"] == "lora/e2e-original" and not config.get("local_data"):
            from utils.paper_resources import manifest

            write_json(path / "paper_sources.json", manifest())
        train_data = subset(raw["train"], config["train_samples"])
        eval_data = subset(raw["validation"], config["eval_samples"])
        write_json(
            path / "data_summary.json",
            dict(
                dataset=config["dataset"],
                local_data=config["local_data"],
                train_count=len(train_data),
                validation_count=len(eval_data),
                train_fingerprint=train_data._fingerprint,
                validation_fingerprint=eval_data._fingerprint,
                protocol=config.get("protocol", "legacy"),
                loss="target only",
                examples=[train_data[i] for i in range(min(3, len(train_data)))],
            ),
        )
        writer = SummaryWriter(log_dir=str(path / "tensorboard"))
        try:
            writer.add_text("configuration", json.dumps(config, indent=2), 0)
            writer.add_text("training_data", (path / "data_summary.json").read_text(), 0)

            def encode(row):
                """将单条 E2E 数据编码为训练输入与仅覆盖目标文本的标签。"""
                return tokenize(
                    row, tokenizer, config["max_length"], config.get("protocol", "legacy")
                )

            train_data = train_data.map(encode, remove_columns=train_data.column_names)
            eval_data = eval_data.map(encode, remove_columns=eval_data.column_names)
            source_lora = (
                config.get("source_training")
                and kind == "lora"
                and config.get("protocol") == "lora_e2e"
            )
            if source_lora:
                from utils.source_training import FullBatchDataset, SourceCollator

                train_data = FullBatchDataset(train_data, config["batch_size"])
                eval_data = FullBatchDataset(eval_data, config["eval_batch_size"])
            # 官方 LoRA GPT-2 仅在 LoRA 输入使用 dropout；FFT 使用原生 dropout。
            model_dropout = 0.0 if kind == "lora" else config["dropout"]
            model = AutoModelForCausalLM.from_pretrained(
                config["model"],
                local_files_only=config["local_files_only"],
                resid_pdrop=model_dropout,
                embd_pdrop=model_dropout,
                attn_pdrop=model_dropout,
            )
            if config["max_length"] > model.config.n_positions:
                raise ValueError("max_length exceeds model context window")
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            if kind == "lora":
                attach(
                    model,
                    config["ranks"],
                    config["alpha"],
                    config["dropout"],
                    shared_dropout=bool(source_lora),
                )
                if config["gradient_checkpointing"]:
                    model.enable_input_require_grads()
            write_json(
                path / "parameters.json",
                dict(
                    total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                ),
            )
            before = {
                name: p.detach().flatten()[:64].cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            args = TrainingArguments(
                output_dir=str(path / "checkpoints"),
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["eval_batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation"],
                learning_rate=config["learning_rate"],
                num_train_epochs=config["epochs"],
                max_steps=config["max_steps"],
                optim="adamw_torch",
                adam_beta1=config.get("adam_beta1", 0.9),
                adam_beta2=config.get("adam_beta2", 0.999),
                adam_epsilon=config.get("adam_epsilon", 1e-8),
                max_grad_norm=config.get("max_grad_norm", 1.0),
                lr_scheduler_type="linear",
                weight_decay=config["weight_decay"],
                warmup_steps=config["warmup_steps"],
                label_smoothing_factor=config["label_smoothing"],
                logging_steps=config["logging_steps"],
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=None,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                fp16=config["precision"] == "fp16",
                bf16=config["precision"] == "bf16",
                gradient_checkpointing=config["gradient_checkpointing"],
                gradient_checkpointing_kwargs={"use_reentrant": False},
                seed=config["seed"],
                data_seed=config["seed"],
                disable_tqdm=True,
            )
            trainer_class = AdapterTrainer if kind == "lora" else SourceTrainer
            extras = {"run_config": config}
            trainer = trainer_class(
                model=model,
                args=args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=eval_data,
                data_collator=SourceCollator(config["max_length"])
                if source_lora
                else DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, padding=True, label_pad_token_id=-100
                ),
                callbacks=[TensorBoardCallback(tb_writer=writer), Progress(path)],
                **extras,
            )
            try:
                result = trainer.train(resume_from_checkpoint=checkpoint)
                if (path / "stop.request.json").exists():
                    raise KeyboardInterrupt("用户停止训练；已保存 step 边界 checkpoint。")
                updated = any(
                    not torch.equal(before[name], p.detach().flatten()[:64].cpu())
                    for name, p in model.named_parameters()
                    if name in before
                )
                if not updated:
                    raise RuntimeError(
                        "No sampled trainable weights changed; "
                        "optimizer updates may have been skipped."
                    )
                trainer.save_model(str(path / "final"))
                tokenizer.save_pretrained(path / "final")
                write_json(
                    path / "result.json",
                    dict(
                        metrics=result.metrics,
                        best_checkpoint=trainer.state.best_model_checkpoint,
                        best_eval_loss=trainer.state.best_metric,
                        final=str(path / "final"),
                        weight_update_verified=updated,
                    ),
                )
            finally:
                write_json(path / "train_log.json", trainer.state.log_history)
                writer.flush()
        finally:
            writer.close()
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return path
