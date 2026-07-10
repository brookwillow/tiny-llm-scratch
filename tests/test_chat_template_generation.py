import json
import tempfile
import unittest
from pathlib import Path

from tokenizer.chat_template import encode_chat_generation_prompt


class FakeTokenizer:
    vocab = {"<|im_start|>": 100, "<|im_end|>": 101}

    def encode(self, text):
        return [ord(char) for char in text]


class ChatTemplateGenerationTests(unittest.TestCase):
    def test_appends_an_assistant_header_for_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "tokenizer_config.json"
            config_path.write_text(
                json.dumps({"special_tokens": ["<|im_start|>", "<|im_end|>"]}),
                encoding="utf-8",
            )

            token_ids = encode_chat_generation_prompt(
                [{"role": "user", "content": "hi"}],
                FakeTokenizer(),
                config_path=config_path,
            )

        self.assertEqual(
            token_ids,
            [100, *map(ord, "user\nhi"), 101, ord("\n"), 100, *map(ord, "assistant\n")],
        )
