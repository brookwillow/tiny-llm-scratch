import unittest
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.chat_template import apply_chat_template, encode_chat_messages, load_chat_template_config


ROOT = Path(__file__).resolve().parents[1]


class ChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer(
            ROOT / "tokenizer" / "vocab.json",
            ROOT / "tokenizer" / "merges.txt",
        )

    def test_loads_special_tokens_from_tokenizer_config(self):
        config = load_chat_template_config(ROOT / "tokenizer" / "tokenizer_config.json")

        self.assertEqual(config.im_start, "<|im_start|>")
        self.assertEqual(config.im_end, "<|im_end|>")

    def test_apply_chat_template_formats_messages(self):
        text = apply_chat_template(
            [
                {"role": "system", "content": "你是助手。"},
                {"role": "user", "content": "你好"},
            ],
            add_generation_prompt=True,
            config_path=ROOT / "tokenizer" / "tokenizer_config.json",
        )

        self.assertIn("<|im_start|>system\n你是助手。<|im_end|>\n", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))

    def test_encode_chat_messages_masks_only_assistant_response(self):
        encoded = encode_chat_messages(
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什么可以帮你？"},
            ],
            self.tokenizer,
            config_path=ROOT / "tokenizer" / "tokenizer_config.json",
        )

        self.assertEqual(len(encoded.input_ids), len(encoded.loss_mask))
        self.assertGreater(sum(encoded.loss_mask), 0)
        self.assertEqual(encoded.input_ids[0], self.tokenizer.vocab["<|im_start|>"])
        self.assertEqual(encoded.input_ids[-2], self.tokenizer.vocab["<|im_end|>"])


if __name__ == "__main__":
    unittest.main()
