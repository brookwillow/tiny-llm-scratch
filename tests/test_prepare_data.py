import json
import tempfile
import unittest
from array import array
from pathlib import Path

from data.prepare_data import tokenize_jsonl_to_bins


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class PrepareDataTests(unittest.TestCase):
    def test_tokenize_jsonl_streams_to_train_and_val_bins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_path = temp_path / "sample.jsonl"
            train_path = temp_path / "train.bin"
            val_path = temp_path / "val.bin"
            rows = [{"text": "ab"}, {"text": "cd"}, {"text": "ef"}, {"text": "gh"}]
            raw_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            stats = tokenize_jsonl_to_bins(
                raw_file=raw_path,
                train_bin=train_path,
                val_bin=val_path,
                tokenizer=FakeTokenizer(),
                val_ratio=0.25,
                max_lines=None,
                num_workers=1,
                chunk_size=2,
                log_interval=0,
                total_lines=4,
            )

            train = array("H")
            val = array("H")
            with train_path.open("rb") as f:
                train.fromfile(f, train_path.stat().st_size // train.itemsize)
            with val_path.open("rb") as f:
                val.fromfile(f, val_path.stat().st_size // val.itemsize)

            self.assertEqual(stats["processed_lines"], 4)
            self.assertEqual(stats["train_tokens"], 6)
            self.assertEqual(stats["val_tokens"], 2)
            self.assertEqual(train.tolist(), [ord("a"), ord("b"), ord("c"), ord("d"), ord("e"), ord("f")])
            self.assertEqual(val.tolist(), [ord("g"), ord("h")])


if __name__ == "__main__":
    unittest.main()
