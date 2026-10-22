"""Independent padding policy, including inference for historical run snapshots."""


def resolve_padding(config, kind):
    """Missing historical fields preserve the collator selected before this option existed."""
    mode = config.get("padding_mode")
    if mode is None:
        source_lora = (
            kind == "lora"
            and config.get("source_training")
            and config.get("protocol") == "lora_e2e"
        )
        mode = "fixed" if source_lora else "dynamic"
    if mode not in ("fixed", "dynamic"):
        raise ValueError("padding_mode must be fixed or dynamic")
    return mode


def training_collator(config, kind, tokenizer):
    """Change only padding, retaining author LoRA padding IDs and target masks."""
    from transformers import DataCollatorForSeq2Seq

    from utils.source_training import SourceCollator

    mode = resolve_padding(config, kind)
    if kind == "lora" and config.get("source_training") and config.get("protocol") == "lora_e2e":
        return SourceCollator(config["max_length"], padding_mode=mode)
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length" if mode == "fixed" else True,
        max_length=config["max_length"] if mode == "fixed" else None,
        label_pad_token_id=-100,
    )
