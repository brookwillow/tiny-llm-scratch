import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.model_config import add_model_config_arg, resolve_model_config


def select_context_window(token_ids: list[int], context_length: int) -> list[int]:
    return token_ids[-context_length:]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained tiny LLM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="你好")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat", action="store_true", help="Format the prompt with the SFT chat template.")
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--vocab", type=str, default=str(ROOT_DIR / "tokenizer" / "vocab.json"))
    parser.add_argument("--merges", type=str, default=str(ROOT_DIR / "tokenizer" / "merges.txt"))
    parser.add_argument("--tokenizer_config", type=str, default=str(ROOT_DIR / "tokenizer" / "tokenizer_config.json"))
    parser.add_argument("--device", type=str, default=None)

    add_model_config_arg(parser, default=str(ROOT_DIR / "config" / "model.json"))
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    return parser.parse_args()


def load_model(args):
    import torch
    from model.model import Model, ModelConfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
        device=device,
    )
    model = Model(config=config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def sample_next_token(logits, temperature: float, top_k: int):
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(model, token_ids: list[int], max_new_tokens: int, context_length: int, device: str, temperature: float, top_k: int, stop_token_ids: set[int] | None = None):
    import torch

    if not token_ids:
        raise ValueError("prompt must encode to at least one token")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = select_context_window(token_ids, context_length)
            x = torch.tensor([context_ids], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1, :]
            next_id = sample_next_token(next_logits, temperature=temperature, top_k=top_k)
            if stop_token_ids and next_id in stop_token_ids:
                break
            token_ids.append(next_id)

    return token_ids


def main():
    args = parse_args()
    resolve_model_config(args)

    import torch
    from tokenizer.bpe_tokenizer import BPETokenizer

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer(args.vocab, args.merges)
    model, device = load_model(args)

    if args.chat:
        from tokenizer.chat_template import encode_chat_generation_prompt, load_chat_template_config

        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        token_ids = encode_chat_generation_prompt(messages, tokenizer, config_path=args.tokenizer_config)
        prompt_length = len(token_ids)
        chat_config = load_chat_template_config(args.tokenizer_config)
        stop_token_ids = {tokenizer.vocab[chat_config.im_end]}
    else:
        token_ids = tokenizer.encode(args.prompt)
        prompt_length = 0
        stop_token_ids = None

    token_ids = generate(
        model=model,
        token_ids=token_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_token_ids=stop_token_ids,
    )
    text = tokenizer.decode(token_ids[prompt_length:])
    print(text)


if __name__ == "__main__":
    main()
