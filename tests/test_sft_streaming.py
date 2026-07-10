import json
import tempfile
import unittest
from pathlib import Path

from train.train_sft import JsonlSFTDataset


class JsonlSFTDatasetTests(unittest.TestCase):
    def test_indexes_jsonl_without_tokenizing_all_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "sft.jsonl"
            data_path.write_text(
                "".join(
                    json.dumps({"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i)}]})
                    + "\n"
                    for i in range(3)
                ),
                encoding="utf-8",
            )

            dataset = JsonlSFTDataset(data_path, max_samples=None)

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.get_sample(1)["messages"][0]["content"], "1")


if __name__ == "__main__":
    unittest.main()
