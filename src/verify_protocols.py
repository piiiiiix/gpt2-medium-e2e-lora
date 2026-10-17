"""第 8 节回归检查：输入 token、优化器数值、生成协议与历史配置兼容。"""

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from config import EVAL_DEFAULTS, FFT_DEFAULTS, MODEL_NAME, OUTPUT_ROOT, TRAIN_DEFAULTS
from utils.cli import arguments
from utils.data import context, prompt, tokenize
from utils.protocols import clean_prediction, model_protocol, resolve_generation, save_protocol
from utils.source_optimizer import SourceAdamW


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        cls.row = dict(
            meaning_representation="name[The Vaults], eatType[pub], priceRange[high]",
            target="The Vaults is a pub.",
        )

    def test_encoding(self):
        tok = self.tokenizer
        ctx = "name : The Vaults | Type : pub | price : high"
        self.assertEqual(context(self.row), ctx)

        def enc(text):
            return tok(text, add_special_tokens=False)["input_ids"]

        for protocol, prefix, target in [
            ("lora_e2e", enc(ctx) + [50256], enc(" " + self.row["target"]) + [50256]),
            (
                "prefix_ft",
                enc(" " + ctx + " ") + [50256],
                enc(" " + self.row["target"] + " ") + [50256],
            ),
            (
                "legacy",
                enc("MR: " + self.row["meaning_representation"] + "\nText:"),
                enc(" " + self.row["target"]) + [50256],
            ),
        ]:
            actual = tokenize(self.row, tok, 512, protocol)
            self.assertEqual(actual["input_ids"], prefix + target)
            self.assertEqual(actual["labels"], [-100] * len(prefix) + target)
            self.assertEqual(enc(prompt(self.row, protocol, tok)), prefix)
        # 官方原始 context 逐字符保留，避免二次转换。
        self.assertEqual(context({"context": ctx + " "}), ctx + " ")
        with self.assertRaises(ValueError):
            tokenize(self.row, tok, 2, "lora_e2e")

    def test_defaults(self):
        self.assertEqual(
            (TRAIN_DEFAULTS["max_grad_norm"], TRAIN_DEFAULTS["adam_epsilon"]), (0.0, 1e-6)
        )
        self.assertEqual((FFT_DEFAULTS["max_grad_norm"], FFT_DEFAULTS["adam_epsilon"]), (1.0, 1e-8))
        self.assertEqual(FFT_DEFAULTS["warmup_steps"], 0)
        self.assertEqual((FFT_DEFAULTS["learning_rate"], FFT_DEFAULTS["batch_size"]), (5e-5, 10))

    def test_optimizer_numeric_and_resume(self):
        # 独立标量公式，包含近零梯度（能区分 epsilon 放置位置）和非零衰减。
        p = torch.nn.Parameter(torch.tensor([0.75], dtype=torch.float64))
        opt = SourceAdamW([p], lr=0.002, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        expected, m, v = 0.75, 0.0, 0.0
        for t, grad in enumerate([1e-9, -0.3, 0.2], 1):
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            expected -= 0.002 * math.sqrt(1 - 0.999**t) / (1 - 0.9**t) * m / (math.sqrt(v) + 1e-6)
            expected *= 1 - 0.002 * 0.01
            p.grad = torch.tensor([grad], dtype=p.dtype)
            opt.step()
            self.assertAlmostEqual(p.item(), expected, places=14)
        q = torch.nn.Parameter(p.detach().clone())
        resumed = SourceAdamW([q], lr=1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        resumed.load_state_dict(deepcopy(opt.state_dict()))
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
        opt.step()
        resumed.step()
        torch.testing.assert_close(p, q)

    def test_profiles_and_cleaning(self):
        cfg = deepcopy(EVAL_DEFAULTS)
        self.assertEqual(resolve_generation(cfg)["eos_token_id"], [50256, 628])
        cfg["generation_profile"] = "prefix_ft"
        self.assertEqual(resolve_generation(cfg)["num_beams"], 5)
        self.assertEqual(resolve_generation(cfg)["max_new_tokens"], 100)
        cfg.update(num_beams=2, max_new_tokens=7)
        self.assertEqual(resolve_generation(cfg)["num_beams"], 2)
        self.assertEqual(clean_prediction(" first\n\nsecond", "lora_e2e"), "first")
        self.assertEqual(clean_prediction(" first<|endoftext|>second", "prefix_ft"), "first")

    def test_saved_and_legacy_models(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT / "smoke") as tmp:
            spec = dict(type="fft", path=tmp)
            self.assertEqual(model_protocol(spec), "legacy")
            cfg = {**EVAL_DEFAULTS, "model": spec}
            self.assertEqual(resolve_generation(cfg)["protocol"], "legacy")
            save_protocol(tmp, FFT_DEFAULTS)
            self.assertEqual(resolve_generation(cfg)["profile"], "prefix_ft")

    def test_queued_legacy_config(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT / "smoke") as tmp:
            path = Path(tmp)
            cfg = {
                k: v
                for k, v in TRAIN_DEFAULTS.items()
                if k
                not in (
                    "protocol",
                    "optimizer",
                    "adam_beta1",
                    "adam_beta2",
                    "adam_epsilon",
                    "max_grad_norm",
                )
            }
            (path / "config.json").write_text(json.dumps(cfg))
            (path / "status.json").write_text('{"status":"queued"}')
            with patch(
                "sys.argv", ["train", "--config", str(path / "config.json"), "--run-dir", str(path)]
            ):
                resolved, _ = arguments(TRAIN_DEFAULTS, "lora")
            self.assertEqual(resolved["protocol"], "legacy")
            self.assertEqual(resolved["optimizer"], "adamw_torch")
            self.assertEqual(resolved["max_grad_norm"], 1.0)


if __name__ == "__main__":
    unittest.main()
