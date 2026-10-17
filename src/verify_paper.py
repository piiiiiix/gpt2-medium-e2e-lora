"""Regression checks against pinned official sources; artifacts stay under outputs/smoke."""

import ast
import importlib.util
import json
import os
import time
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from config import MODEL_NAME, OUTPUT_ROOT
from utils.data import original_data, tokenize
from utils.evaluation import check_coverage
from utils.paper_resources import fetch, manifest
from utils.runs import write_json
from utils.source_beam import generate
from utils.source_training import source_loss


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = OUTPUT_ROOT / "smoke/paper_alignment/regression"
        cls.folder.mkdir(parents=True, exist_ok=True)
        cls.data = original_data(True)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        cls.sources = {
            name: fetch(item, OUTPUT_ROOT / "resources/paper_reference" / name)
            for name, item in manifest()["verification"].items()
        }

    def test_data_and_incomplete_rescore(self):
        self.assertEqual(
            {k: len(v) for k, v in self.data.items()},
            dict(train=42061, validation=4672, validation_generation=547, test=630),
        )
        self.assertEqual(sum(len(r["references"]) for r in self.data["test"]), 4693)
        rows = [dict(r, index=i, prediction=r["target"]) for i, r in enumerate(self.data["test"])]
        check_coverage(rows, self.data["test"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            check_coverage(rows[:613], self.data["test"])
        broken = deepcopy(rows)
        broken[0]["references"] = broken[0]["references"][:1]
        with self.assertRaisesRegex(ValueError, "reference"):
            check_coverage(broken, self.data["test"])
        with self.assertRaisesRegex(ValueError, "order"):
            check_coverage(rows[::-1], self.data["test"])

    def test_bpe_against_author_encoder(self):
        spec = importlib.util.spec_from_file_location("author_encoder", self.sources["encoder.py"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = json.loads(self.tokenizer.backend_tokenizer.to_str())["model"]
        merges = [tuple(m.split()) if isinstance(m, str) else tuple(m) for m in backend["merges"]]
        encoder = module.Encoder(backend["vocab"], merges)
        for split in ("train", "validation"):
            for row in self.data[split].select(range(32)):
                prefix = encoder.encode(row["context"])[0] + [50256]
                target = encoder.encode(" " + row["target"])[0] + [50256]
                actual = tokenize(row, self.tokenizer, 512, "lora_e2e")
                self.assertEqual(actual["input_ids"], prefix + target)
                self.assertEqual(actual["labels"], [-100] * len(prefix) + target)

    def test_source_loss_and_validation(self):
        torch.manual_seed(10)
        logits = torch.randn(2, 5, 9, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 5, 6, 7, 8]])
        for smooth in (0.0, 0.1):
            # Independent scalar target-token calculation, including the source denominator.
            terms = []
            for row in range(2):
                for pos in range(1, 5):
                    if labels[row, pos] != -100:
                        probs = logits[row, pos - 1].softmax(-1)
                        terms.append(
                            -(1 - smooth) * probs[labels[row, pos]].log()
                            - smooth * probs.log().mean()
                        )
            expected = torch.stack(terms).sum() / (len(terms) + 0.0001)
            torch.testing.assert_close(source_loss(logits, labels, smooth), expected)
        self.assertNotEqual(
            source_loss(logits, labels, 0).item(), source_loss(logits, labels, 0.1).item()
        )

    def test_beam_against_author_loop(self):
        # Execute checked source functions without its CLI/distributed initialization.
        names = {
            "_calc_banned_ngram_tokens",
            "_postprocess_next_token_scores",
            "_add_beam_candidate",
            "beam",
        }
        tree = ast.parse(self.sources["gpt2_beam.py"].read_text(encoding="utf-8"))
        module = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
            type_ignores=[],
        )
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=23,
                n_positions=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                attn_pdrop=0,
                embd_pdrop=0,
            )
        ).eval()
        ids = torch.tensor([[1, 2, 3, 2]])

        class Bridge:
            def eval(self):
                model.eval()

            def __call__(self, tokens, past=None, len_past=None):
                result = model(tokens, past_key_values=past, use_cache=True)
                return result.logits, result.past_key_values

        def reorder(cache, indices):
            cache.reorder_cache(indices)
            return cache

        for eos, ngram in ((22, 2), (2, 1), (3, 0), ([22, 2], 2)):
            args = SimpleNamespace(
                device="cpu",
                beam=3,
                eval_len=8,
                length_penalty=0.9,
                repetition_penalty=1.0,
                no_repeat_ngram_size=ngram,
                min_length=0,
                eos_token_id=eos if isinstance(eos, list) else [eos],
                rank=0,
                work_dir=str(self.folder),
                output_file="beam.jsonl",
            )
            namespace = dict(
                torch=torch,
                Tensor=torch.Tensor,
                F=torch.nn.functional,
                time=time,
                os=os,
                json=json,
                args=args,
                _reorder_cache=reorder,
                distributed_gather=lambda args, x: x,
                distributed_sync=lambda args: None,
            )
            exec(compile(module, "pinned_gpt2_beam.py", "exec"), namespace)
            namespace["beam"](
                Bridge(), [dict(id=torch.tensor([0]), query=ids, query_len=torch.tensor([4]))], args
            )
            expected = json.loads((self.folder / "beam.jsonl").read_text())["predict"]
            with torch.no_grad():
                actual = generate(
                    model,
                    ids,
                    dict(
                        num_beams=3,
                        max_new_tokens=8,
                        eos_token_id=eos,
                        no_repeat_ngram_size=ngram,
                        length_penalty=0.9,
                    ),
                )[0, 4:].tolist()
            endings = eos if isinstance(eos, list) else [eos]
            first_eos = next((i for i, token in enumerate(expected) if token in endings), None)
            if first_eos is not None:
                expected = expected[: first_eos + 1]
            self.assertEqual(actual, expected)

    def test_transient_file_lock(self):
        target = self.folder / "atomic.json"
        original = os.replace
        calls = []

        def intermittent(src, dst):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("simulated Windows sharing lock")
            original(src, dst)

        with patch("utils.runs.os.replace", intermittent):
            write_json(target, {"complete": True})
        self.assertEqual(json.loads(target.read_text()), {"complete": True})
        self.assertEqual(len(calls), 3)
        self.assertFalse(list(self.folder.glob("atomic.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
