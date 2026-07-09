import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.model_config import add_model_config_arg, resolve_model_config


ROLE_MAPPING = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "tool": "tool",
}


def normalize_sft_messages(sample: dict) -> list[dict[str, str]]:
    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        return [_normalize_message(message) for message in messages]

    if "conversations" in sample:
        conversations = sample["conversations"]
        if not isinstance(conversations, list):
            raise ValueError("conversations must be a list")
        return [_normalize_conversation_message(message) for message in conversations]

    if "instruction" in sample and ("output" in sample or "response" in sample):
        instruction = str(sample.get("instruction", "")).strip()
        extra_input = str(sample.get("input", "")).strip()
        answer = str(sample.get("output", sample.get("response", "")))
        user_content = instruction if not extra_input else f"{instruction}\n{extra_input}"
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]

    raise ValueError("unsupported SFT sample format")


def _normalize_message(message: dict) -> dict[str, str]:
    role = ROLE_MAPPING.get(str(message.get("role", "")).lower())
    if role is None:
        raise ValueError(f"unsupported role: {message.get('role')!r}")
    return {"role": role, "content": str(message.get("content", ""))}


def _normalize_conversation_message(message: dict) -> dict[str, str]:
    raw_role = message.get("from", message.get("role", ""))
    role = ROLE_MAPPING.get(str(raw_role).lower())
    if role is None:
        raise ValueError(f"unsupported conversation role: {raw_role!r}")
    content = message.get("value", message.get("content", ""))
    return {"role": role, "content": str(content)}


def build_sft_features(
    input_ids: list[int],
    loss_mask: list[int],
    context_length: int,
    pad_id: int,
) -> dict[str, list[int] | list[float]]:
    if len(input_ids) != len(loss_mask):
        raise ValueError("input_ids and loss_mask must have the same length")
    if len(input_ids) < 2:
        raise ValueError("SFT sample must contain at least two tokens")

    max_tokens = context_length + 1
    if len(input_ids) > max_tokens:
        input_ids = input_ids[-max_tokens:]
        loss_mask = loss_mask[-max_tokens:]

    x = input_ids[:-1]
    y = input_ids[1:]
    target_mask = [float(value) for value in loss_mask[1:]]

    pad_len = context_length - len(x)
    if pad_len > 0:
        x.extend([pad_id] * pad_len)
        y.extend([pad_id] * pad_len)
        target_mask.extend([0.0] * pad_len)

    return {"x": x, "y": y, "loss_mask": target_mask}


def load_sft_feature_from_sample(
    sample: dict,
    tokenizer,
    tokenizer_config: str,
    context_length: int,
    pad_id: int,
) -> dict[str, list[int] | list[float]]:
    if {"x", "y", "loss_mask"}.issubset(sample):
        feature = {
            "x": [int(value) for value in sample["x"]],
            "y": [int(value) for value in sample["y"]],
            "loss_mask": [float(value) for value in sample["loss_mask"]],
        }
        if len(feature["x"]) != context_length or len(feature["y"]) != context_length:
            raise ValueError("preprocessed SFT x/y length must equal context_length")
        if len(feature["loss_mask"]) != context_length:
            raise ValueError("preprocessed SFT loss_mask length must equal context_length")
        return feature

    from tokenizer.chat_template import encode_chat_messages

    messages = normalize_sft_messages(sample)
    encoded = encode_chat_messages(messages, tokenizer, config_path=tokenizer_config)
    return build_sft_features(encoded.input_ids, encoded.loss_mask, context_length, pad_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tune the tiny LLM on chat JSONL data.")
    add_model_config_arg(parser, default=str(ROOT_DIR / "config" / "model.json"))

    parser.add_argument("--sft_data_path", type=str, required=True)
    parser.add_argument("--valid_sft_data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="train/sft_out")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true")

    parser.add_argument("--vocab", type=str, default=str(ROOT_DIR / "tokenizer" / "vocab.json"))
    parser.add_argument("--merges", type=str, default=str(ROOT_DIR / "tokenizer" / "merges.txt"))
    parser.add_argument("--tokenizer_config", type=str, default=str(ROOT_DIR / "tokenizer" / "tokenizer_config.json"))
    parser.add_argument("--pad_token", type=str, default="<unk>")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--pretrain_data_path", type=str, default=None)
    parser.add_argument("--pretrain_mix_ratio", type=float, default=0.0)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def load_sft_dataset(path: str, tokenizer, tokenizer_config: str, context_length: int, pad_id: int, max_samples: int | None):
    features = []
    skipped = 0
    start_time = time.time()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_samples is not None and len(features) >= max_samples:
                break
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
                feature = load_sft_feature_from_sample(sample, tokenizer, tokenizer_config, context_length, pad_id)
                if sum(feature["loss_mask"]) <= 0:
                    skipped += 1
                    continue
                features.append(feature)
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"跳过第 {line_no} 行: {exc}")

            if line_no % 1000 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                speed = line_no / elapsed
                print(f"  已读取 {line_no} 行, 可训练样本 {len(features)}, 速度 {speed:.1f} lines/s")

    if not features:
        raise ValueError(f"no usable SFT samples loaded from {path}")

    print(f"加载完成: {path}, 可训练样本 {len(features)}, 跳过 {skipped}")
    return features


def get_sft_batch(samples, batch_size: int, device: str):
    import torch

    batch = random.choices(samples, k=batch_size)
    x = torch.tensor([sample["x"] for sample in batch], dtype=torch.long, device=device)
    y = torch.tensor([sample["y"] for sample in batch], dtype=torch.long, device=device)
    mask = torch.tensor([sample["loss_mask"] for sample in batch], dtype=torch.float32, device=device)
    return x, y, mask


def masked_cross_entropy(logits, target, loss_mask):
    import torch

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    target_logits = torch.gather(logits, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    shifted_logits = logits - max_logits
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    per_token_loss = log_sum_exp - target_logits
    mask_sum = loss_mask.sum().clamp_min(1.0)
    return (per_token_loss * loss_mask).sum() / mask_sum


def load_model_state(checkpoint_path: str, model):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def evaluate(model, samples, batch_size: int, device: str, eval_iters: int):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y, mask = get_sft_batch(samples, batch_size, device)
            logits = model(x)
            losses.append(masked_cross_entropy(logits, y, mask).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    resolve_model_config(args)

    import numpy as np
    import torch

    from model.model import Model, ModelConfig
    from tokenizer.bpe_tokenizer import BPETokenizer
    from train.train_pretrain import (
        AdamW,
        clip_gradient_norm,
        get_batch,
        get_lr_cosine_schedule,
        load_checkpoint,
        move_optimizer_state_to_device,
        save_checkpoint,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = BPETokenizer(args.vocab, args.merges)
    if args.pad_token not in tokenizer.vocab:
        raise ValueError(f"pad token {args.pad_token!r} not found in tokenizer vocab")
    pad_id = tokenizer.vocab[args.pad_token]

    train_samples = load_sft_dataset(
        args.sft_data_path,
        tokenizer,
        args.tokenizer_config,
        args.context_length,
        pad_id,
        args.max_samples,
    )
    valid_samples = (
        load_sft_dataset(
            args.valid_sft_data_path,
            tokenizer,
            args.tokenizer_config,
            args.context_length,
            pad_id,
            args.max_samples,
        )
        if args.valid_sft_data_path
        else train_samples
    )

    pretrain_data = None
    if args.pretrain_data_path and args.pretrain_mix_ratio > 0:
        pretrain_data = np.memmap(args.pretrain_data_path, dtype=np.uint16, mode="r")
        print(f"预训练混合数据: {len(pretrain_data)} tokens, mix_ratio={args.pretrain_mix_ratio}")

    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
        device=device,
    )
    model = Model(config=config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if not args.no_resume and os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        model.to(device)
        move_optimizer_state_to_device(optimizer, device)
        print(f"从 SFT checkpoint 恢复: iter={start_iter}")
    elif args.init_checkpoint:
        load_model_state(args.init_checkpoint, model)
        model.to(device)
        print(f"加载预训练权重: {args.init_checkpoint}")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as exc:
            print(f"wandb 初始化失败, 改为本地训练: {exc}")

    print(
        f"SFT 开始: train_samples={len(train_samples)}, valid_samples={len(valid_samples)}, "
        f"device={device}, batch_size={args.batch_size}, context={args.context_length}"
    )

    train_loss = None
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        if pretrain_data is not None and random.random() < args.pretrain_mix_ratio:
            x, y = get_batch(pretrain_data, args.batch_size, args.context_length, device)
            mask = torch.ones_like(y, dtype=torch.float32, device=device)
        else:
            x, y, mask = get_sft_batch(train_samples, args.batch_size, device)

        logits = model(x)
        loss = masked_cross_entropy(logits, y, mask)

        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(), args.max_norm)
        optimizer.step()
        train_loss = loss.item()

        if it % args.log_interval == 0 or it == args.max_iters - 1:
            print(f"Iter {it}: train_loss {train_loss:.4f}, lr {lr:.6g}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": train_loss, "lr": lr, "iter": it + 1})

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            val_loss = evaluate(model, valid_samples, args.batch_size, device, args.eval_iters)
            print(f"Iter {it}: val_loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss, "iter": it + 1})

        if it % args.save_interval == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
