"""Project defaults. Paper values: LoRA v2 appendix D.3 / Table 11."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
MODEL_NAME = 'openai-community/gpt2-medium'
DATASET_NAME = 'GEM/e2e_nlg'
METRICS = ['BLEU', 'NIST', 'METEOR', 'ROUGE-L', 'CIDEr']
TRAIN_DEFAULTS = dict(model=MODEL_NAME, dataset=DATASET_NAME, learning_rate=2e-4,
    epochs=5, batch_size=8, eval_batch_size=8, gradient_accumulation=1,
    weight_decay=0.01, warmup_steps=500, label_smoothing=0.1,
    dropout=0.1, alpha=32, seed=42, precision='fp32', max_length=512,
    gradient_checkpointing=True, logging_steps=10, max_steps=-1,
    train_samples=None, eval_samples=None, local_data=None, local_files_only=False,
    ranks=[dict(q=4, k=0, v=4) for _ in range(24)])
# FFT defaults are a comparison implementation profile, not Table 11 LoRA values.
FFT_DEFAULTS = {**TRAIN_DEFAULTS, 'learning_rate': 2e-5}
EVAL_DEFAULTS = dict(dataset=DATASET_NAME, split='validation', samples=None,
    metrics=METRICS.copy(), max_new_tokens=60, num_beams=10, length_penalty=0.9,
    no_repeat_ngram_size=4, seed=42, local_data=None, local_files_only=False,
    model=dict(type='base', path=MODEL_NAME, name='GPT-2 Medium'), predictions=None)
GPU_INTERVAL_SECONDS = 2.0

def require_lora():
    import sys
    if Path(sys.prefix).name.lower() != 'lora':
        raise RuntimeError(f'Use conda environment lora, current interpreter: {sys.executable}')

def inside_project(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'Path outside medium project: {path}')
    return path
