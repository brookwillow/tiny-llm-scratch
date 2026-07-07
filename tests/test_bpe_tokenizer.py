import unittest
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]


class BPETokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer(
            ROOT / "tokenizer" / "vocab.json",
            ROOT / "tokenizer" / "merges.txt",
        )

    def test_round_trip_common_texts(self):
        samples = [
            "Hello, world!",
            "你好，世界！这是一个测试。",
            "今天 temperature is 28.5°C, humidity is 72%, 风有点大。",
            "def merge_lists(a, b):\n    return sorted(set(a + b))\n",
            "こんにちは、世界！안녕하세요, 세계! Привет, мир! مرحبا بالعالم!",
        ]

        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_empty_string_encodes_to_empty_list(self):
        self.assertEqual(self.tokenizer.encode(""), [])


if __name__ == "__main__":
    unittest.main()
