import json
import tempfile
import unittest
from pathlib import Path

from data.prepare_data import split_dpo_jsonl_to_jsonl


class PrepareDPOTests(unittest.TestCase):
    def test_splits_and_normalizes_dpo_pairs(self):
        samples = [
            {
                "chosen": [
                    {"role": "human", "content": "hello"},
                    {"role": "assistant", "content": "good answer"},
                ],
                "rejected": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "bad answer"},
                ],
            },
            {
                "chosen": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "chosen"},
                ],
                "rejected": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "rejected"},
                ],
            },
            {"chosen": [], "rejected": []},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "dpo.jsonl"
            train_path = temp_path / "dpo_train.jsonl"
            val_path = temp_path / "dpo_val.jsonl"
            raw_path.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            stats = split_dpo_jsonl_to_jsonl(
                raw_path, train_path, val_path, val_ratio=0.5, max_lines=None, log_interval=0, total_lines=3
            )

            train_sample = json.loads(train_path.read_text(encoding="utf-8"))
            val_sample = json.loads(val_path.read_text(encoding="utf-8"))

        self.assertEqual(stats, {"processed_lines": 2, "train_samples": 1, "val_samples": 1, "skipped_lines": 1})
        self.assertEqual(train_sample["chosen"][0]["role"], "user")
        self.assertEqual(val_sample["rejected"][-1]["content"], "rejected")

