import json
from pathlib import Path


MODEL_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "hidden_size",
    "num_layers",
    "num_heads",
)


def load_model_config(path: str | Path | None) -> dict:
    if path is None:
        return {}

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    unknown_keys = set(config) - set(MODEL_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"unknown model config keys: {sorted(unknown_keys)}")

    return config


def apply_model_config(args, config: dict) -> None:
    for key in MODEL_CONFIG_KEYS:
        if getattr(args, key, None) is None and key in config:
            setattr(args, key, config[key])


def add_model_config_arg(parser, default: str | None = "configs/model_512x6.json") -> None:
    parser.add_argument("--model_config", type=str, default=default)


def resolve_model_config(args) -> dict:
    config = load_model_config(getattr(args, "model_config", None))
    apply_model_config(args, config)
    return config
