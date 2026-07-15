import json
import tempfile
import unittest
from pathlib import Path

from train.train_dpo import load_dpo_features_from_sample, normalize_dpo_pair


class FakeTokenizer:
    vocab = {"<|im_start|>": 100, "<|im_end|>": 101}

    def encode(self, text):
        return [ord(char) for char in text]


class DPODataTests(unittest.TestCase):
    def test_normalizes_each_preference_branch(self):
        chosen, rejected = normalize_dpo_pair(
            {
                "chosen": [
                    {"role": "human", "content": "hello"},
                    {"role": "assistant", "content": "good"},
                ],
                "rejected": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "bad"},
                ],
            }
        )

        self.assertEqual(chosen[0], {"role": "user", "content": "hello"})
        self.assertEqual(rejected[-1], {"role": "assistant", "content": "bad"})

    def test_rejects_branches_without_final_assistant(self):
        with self.assertRaisesRegex(ValueError, "end with an assistant"):
            normalize_dpo_pair(
                {
                    "chosen": [{"role": "user", "content": "question"}],
                    "rejected": [{"role": "assistant", "content": "answer"}],
                }
            )

    def test_builds_masked_features_for_both_responses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "tokenizer_config.json"
            config_path.write_text(
                json.dumps({"special_tokens": ["<|im_start|>", "<|im_end|>"]}), encoding="utf-8"
            )
            chosen, rejected = load_dpo_features_from_sample(
                {
                    "chosen": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "good"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "bad"},
                    ],
                },
                FakeTokenizer(),
                str(config_path),
                context_length=32,
                pad_id=0,
            )

        self.assertEqual(len(chosen["x"]), 32)
        self.assertEqual(len(rejected["y"]), 32)
        self.assertGreater(sum(chosen["loss_mask"]), 0)
        self.assertGreater(sum(rejected["loss_mask"]), 0)
