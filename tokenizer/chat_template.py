import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "tokenizer_config.json"


@dataclass(frozen=True)
class ChatTemplateConfig:
    im_start: str
    im_end: str


@dataclass(frozen=True)
class EncodedChat:
    input_ids: list[int]
    loss_mask: list[int]


def load_chat_template_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ChatTemplateConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)

    special_tokens = set(config.get("special_tokens", []))
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    missing = [token for token in (im_start, im_end) if token not in special_tokens]
    if missing:
        raise ValueError(f"tokenizer config missing chat special tokens: {missing}")

    return ChatTemplateConfig(im_start=im_start, im_end=im_end)


def apply_chat_template(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> str:
    config = load_chat_template_config(config_path)
    parts: list[str] = []

    for message in messages:
        role = validate_role(message["role"])
        content = message.get("content", "")
        parts.append(f"{config.im_start}{role}\n")
        parts.append(content)
        parts.append(f"{config.im_end}\n")

    if add_generation_prompt:
        parts.append(f"{config.im_start}assistant\n")

    return "".join(parts)


def encode_chat_messages(
    messages: list[dict[str, str]],
    tokenizer,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> EncodedChat:
    config = load_chat_template_config(config_path)
    im_start_id = tokenizer.vocab[config.im_start]
    im_end_id = tokenizer.vocab[config.im_end]

    input_ids: list[int] = []
    loss_mask: list[int] = []

    for message in messages:
        role = validate_role(message["role"])
        content = message.get("content", "")

        header_ids = [im_start_id] + tokenizer.encode(f"{role}\n")
        content_ids = tokenizer.encode(content)
        end_ids = [im_end_id] + tokenizer.encode("\n")

        message_ids = header_ids + content_ids + end_ids
        input_ids.extend(message_ids)

        if role == "assistant":
            loss_mask.extend([0] * len(header_ids))
            loss_mask.extend([1] * len(content_ids))
            loss_mask.extend([1] * len(end_ids))
        else:
            loss_mask.extend([0] * len(message_ids))

    return EncodedChat(input_ids=input_ids, loss_mask=loss_mask)


def encode_chat_generation_prompt(
    messages: list[dict[str, str]],
    tokenizer,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[int]:
    """Encode chat history and append the assistant header to be completed."""
    config = load_chat_template_config(config_path)
    im_start_id = tokenizer.vocab[config.im_start]
    im_end_id = tokenizer.vocab[config.im_end]

    input_ids: list[int] = []
    for message in messages:
        role = validate_role(message["role"])
        content = message.get("content", "")
        input_ids.extend([im_start_id])
        input_ids.extend(tokenizer.encode(f"{role}\n"))
        input_ids.extend(tokenizer.encode(content))
        input_ids.extend([im_end_id])
        input_ids.extend(tokenizer.encode("\n"))

    input_ids.extend([im_start_id])
    input_ids.extend(tokenizer.encode("assistant\n"))
    return input_ids


def validate_role(role: str) -> str:
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported chat role: {role!r}")
    return role
