import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from config.model_config import apply_model_config, load_model_config


class ModelConfigTests(unittest.TestCase):
    def test_load_model_config_reads_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "model.json"
            config_path.write_text(
                json.dumps(
                    {
                        "vocab_size": 6400,
                        "context_length": 256,
                        "hidden_size": 512,
                        "num_layers": 6,
                        "num_heads": 8,
                    }
                ),
                encoding="utf-8",
            )

            config = load_model_config(config_path)

            self.assertEqual(config["hidden_size"], 512)
            self.assertEqual(config["num_layers"], 6)

    def test_apply_model_config_keeps_explicit_cli_values(self):
        args = Namespace(
            vocab_size=None,
            context_length=128,
            hidden_size=None,
            num_layers=None,
            num_heads=None,
        )
        config = {
            "vocab_size": 6400,
            "context_length": 256,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
        }

        apply_model_config(args, config)

        self.assertEqual(args.vocab_size, 6400)
        self.assertEqual(args.context_length, 128)
        self.assertEqual(args.hidden_size, 512)
        self.assertEqual(args.num_layers, 6)
        self.assertEqual(args.num_heads, 8)


if __name__ == "__main__":
    unittest.main()
