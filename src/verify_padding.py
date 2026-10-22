"""Padding policy regression: masks/gradients, old snapshots, and submitted UI config."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from config import FFT_DEFAULTS, MODEL_NAME, OUTPUT_ROOT, TRAIN_DEFAULTS
from utils.cli import arguments
from utils.padding import resolve_padding, training_collator
from utils.training import SourceTrainer


class PaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        cls.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_shapes_masks_and_gradients(self):
        rows = [
            dict(input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4]),
            dict(input_ids=[1, 5, 6], labels=[-100, 5, 6]),
        ]
        for kind, defaults in (("lora", TRAIN_DEFAULTS), ("fft", FFT_DEFAULTS)):
            config = {**deepcopy(defaults), "max_length": 8}
            torch.manual_seed(42)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(self.tokenizer),
                    n_layer=1,
                    n_head=1,
                    n_embd=8,
                    n_positions=8,
                    resid_pdrop=0,
                    attn_pdrop=0,
                    embd_pdrop=0,
                )
            ).train()
            trainer = object.__new__(SourceTrainer)
            trainer.run_config = config
            losses, grads = [], []
            for mode, length in (("fixed", 8), ("dynamic", 4)):
                batch = training_collator({**config, "padding_mode": mode}, kind, self.tokenizer)(
                    deepcopy(rows)
                )
                self.assertEqual(tuple(batch["input_ids"].shape), (2, length))
                self.assertEqual(batch["labels"].ne(-100).sum().item(), 4)
                self.assertTrue(batch["labels"][1, 3:].eq(-100).all())
                self.assertTrue(
                    batch["input_ids"][1, 3:]
                    .eq(0 if kind == "lora" else self.tokenizer.pad_token_id)
                    .all()
                )
                model.zero_grad()
                loss = trainer.compute_loss(model, batch)
                loss.backward()
                losses.append(loss.detach())
                grads.append(model.transformer.wte.weight.grad.clone())
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)

    def test_defaults_and_historical_queue(self):
        OUTPUT_ROOT.joinpath("smoke").mkdir(exist_ok=True)
        self.assertEqual(TRAIN_DEFAULTS["padding_mode"], "fixed")
        self.assertEqual(FFT_DEFAULTS["padding_mode"], "dynamic")
        for kind, defaults, expected in (
            ("lora", TRAIN_DEFAULTS, "fixed"),
            ("fft", FFT_DEFAULTS, "dynamic"),
        ):
            config = deepcopy(defaults)
            config.pop("padding_mode")
            self.assertEqual(resolve_padding(config, kind), expected)
            for protocol, mode in ((config["protocol"], expected), ("legacy", "dynamic")):
                config["protocol"] = protocol
                with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT / "smoke") as tmp:
                    path = Path(tmp)
                    (path / "config.json").write_text(json.dumps(config))
                    (path / "status.json").write_text('{"status":"queued"}')
                    with patch(
                        "sys.argv",
                        ["train", "--config", str(path / "config.json"), "--run-dir", str(path)],
                    ):
                        loaded, _ = arguments(defaults, kind)
                    self.assertEqual(loaded["padding_mode"], mode)
        with self.assertRaises(ValueError):
            resolve_padding({"padding_mode": "bad"}, "lora")

    def test_ui_toggle_submits_independent_policy(self):
        from streamlit.testing.v1 import AppTest

        with (
            patch("utils.training_pool.entries", return_value=[]),
            patch("utils.training_pool.add") as add,
        ):
            app = AppTest.from_file(
                str(Path(__file__).resolve().with_name("app.py")), default_timeout=30
            ).run()
            self.assertFalse(app.exception)
            profile = "论文默认 / Prefix-Tuning FT 默认"
            for kind, expected_default, new_dynamic, length in (
                ("lora", False, True, 256),
                ("fft", True, False, 384),
            ):
                app.segmented_control(key="train_kind").set_value(kind).run()
                toggle = app.toggle(key=f"dynamic_padding_{kind}_{profile}")
                self.assertEqual(toggle.value, expected_default)
                toggle.set_value(new_dynamic)
                app.number_input(key=f"max_length_{kind}_{profile}").set_value(length)
                next(b for b in app.button if b.label == "加入待训练池").click().run()
                self.assertFalse(app.exception)
                args = add.call_args.args
                self.assertEqual(args[0], kind)
                self.assertEqual(args[1]["padding_mode"], "dynamic" if new_dynamic else "fixed")
                self.assertEqual(args[1]["max_length"], length)
                self.assertEqual(args[1]["source_training"], True)
                self.assertEqual(args[1]["protocol"], "lora_e2e" if kind == "lora" else "prefix_ft")
                self.assertEqual(args[1]["label_smoothing"], 0.1 if kind == "lora" else 0)


if __name__ == "__main__":
    unittest.main()
