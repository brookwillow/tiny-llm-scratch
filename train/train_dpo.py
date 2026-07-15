"""Direct Preference Optimization training for the tiny LLM."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.model_config import add_model_config_arg, resolve_model_config
from train.train_sft import JsonlDataset, load_model_state, load_sft_feature_from_sample, normalize_sft_messages


def normalize_dpo_pair(sample: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    try:
        chosen = normalize_sft_messages({"messages": sample["chosen"]})
        rejected = normalize_sft_messages({"messages": sample["rejected"]})
    except KeyError as exc:
        raise ValueError("DPO sample must contain chosen and rejected fields") from exc

    if not chosen or not rejected:
        raise ValueError("DPO chosen and rejected conversations must not be empty")
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise ValueError("DPO chosen and rejected conversations must end with an assistant message")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError("DPO chosen and rejected conversations must share the same prompt")
    return chosen, rejected


def load_dpo_features_from_sample(sample: dict, tokenizer, tokenizer_config: str, context_length: int, pad_id: int) -> tuple[dict, dict]:
    chosen, rejected = normalize_dpo_pair(sample)
    chosen_feature = load_sft_feature_from_sample(
        {"messages": chosen}, tokenizer, tokenizer_config, context_length, pad_id
    )
    rejected_feature = load_sft_feature_from_sample(
        {"messages": rejected}, tokenizer, tokenizer_config, context_length, pad_id
    )
    if sum(chosen_feature["loss_mask"]) <= 0 or sum(rejected_feature["loss_mask"]) <= 0:
        raise ValueError("DPO chosen and rejected responses must contain trainable tokens")
    return chosen_feature, rejected_feature


def get_dpo_batch(dataset: JsonlDataset, tokenizer, tokenizer_config: str, context_length: int, pad_id: int, batch_size: int, device: str):
    import torch

    chosen_features = []
    rejected_features = []
    attempts = 0
    max_attempts = max(batch_size * 10, 100)
    while len(chosen_features) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            chosen, rejected = load_dpo_features_from_sample(
                dataset.random_sample(), tokenizer, tokenizer_config, context_length, pad_id
            )
        except Exception:
            continue
        chosen_features.append(chosen)
        rejected_features.append(rejected)

    if len(chosen_features) < batch_size:
        raise RuntimeError(f"unable to build DPO batch: got {len(chosen_features)} pairs after {attempts} attempts")

    def to_tensor(features, key, dtype):
        return torch.tensor([feature[key] for feature in features], dtype=dtype, device=device)

    return (
        to_tensor(chosen_features, "x", torch.long),
        to_tensor(chosen_features, "y", torch.long),
        to_tensor(chosen_features, "loss_mask", torch.float32),
        to_tensor(rejected_features, "x", torch.long),
        to_tensor(rejected_features, "y", torch.long),
        to_tensor(rejected_features, "loss_mask", torch.float32),
    )


def masked_sequence_log_probs(logits, target, loss_mask):
    import torch

    token_log_probs = torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * loss_mask).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float):
    import torch

    policy_log_ratio = policy_chosen - policy_rejected
    reference_log_ratio = reference_chosen - reference_rejected
    reward_margin = beta * (policy_log_ratio - reference_log_ratio)
    loss = -torch.nn.functional.logsigmoid(reward_margin).mean()
    return loss, reward_margin


def evaluate(model, reference_model, dataset, tokenizer, tokenizer_config, context_length, pad_id, batch_size, device, beta, eval_iters):
    import torch

    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for _ in range(eval_iters):
            chosen_x, chosen_y, chosen_mask, rejected_x, rejected_y, rejected_mask = get_dpo_batch(
                dataset, tokenizer, tokenizer_config, context_length, pad_id, batch_size, device
            )
            policy_chosen = masked_sequence_log_probs(model(chosen_x), chosen_y, chosen_mask)
            policy_rejected = masked_sequence_log_probs(model(rejected_x), rejected_y, rejected_mask)
            reference_chosen = masked_sequence_log_probs(reference_model(chosen_x), chosen_y, chosen_mask)
            reference_rejected = masked_sequence_log_probs(reference_model(rejected_x), rejected_y, rejected_mask)
            loss, reward_margin = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
            losses.append(loss.item())
            accuracies.append((reward_margin > 0).float().mean().item())
    model.train()
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def parse_args():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization for the tiny LLM.")
    add_model_config_arg(parser, default=str(ROOT_DIR / "config" / "model.json"))
    parser.add_argument("--dpo_data_path", type=str, required=True)
    parser.add_argument("--valid_dpo_data_path", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, required=True, help="SFT checkpoint used for both policy initialization and reference model.")
    parser.add_argument("--out_dir", type=str, default="train/dpo_out")
    parser.add_argument("--no_resume", action="store_true")

    parser.add_argument("--vocab", type=str, default=str(ROOT_DIR / "tokenizer" / "vocab.json"))
    parser.add_argument("--merges", type=str, default=str(ROOT_DIR / "tokenizer" / "merges.txt"))
    parser.add_argument("--tokenizer_config", type=str, default=str(ROOT_DIR / "tokenizer" / "tokenizer_config.json"))
    parser.add_argument("--pad_token", type=str, default="<unk>")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_model_config(args)

    import torch

    from model.model import Model, ModelConfig
    from tokenizer.bpe_tokenizer import BPETokenizer
    from train.train_pretrain import AdamW, clip_gradient_norm, get_lr_cosine_schedule, load_checkpoint, move_optimizer_state_to_device, save_checkpoint

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = BPETokenizer(args.vocab, args.merges)
    if args.pad_token not in tokenizer.vocab:
        raise ValueError(f"pad token {args.pad_token!r} not found in tokenizer vocab")
    pad_id = tokenizer.vocab[args.pad_token]

    train_dataset = JsonlDataset(args.dpo_data_path, dataset_name="DPO", max_samples=args.max_samples)
    valid_dataset = JsonlDataset(args.valid_dpo_data_path, dataset_name="DPO", max_samples=args.max_samples) if args.valid_dpo_data_path else train_dataset

    config = ModelConfig(args.vocab_size, args.hidden_size, args.num_heads, args.num_layers, args.context_length, device=device)
    model = Model(config=config)
    reference_model = Model(config=copy.deepcopy(config))
    load_model_state(args.init_checkpoint, reference_model)
    reference_model.to(device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if not args.no_resume and os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        model.to(device)
        move_optimizer_state_to_device(optimizer, device)
        print(f"从 DPO checkpoint 恢复: iter={start_iter}")
    else:
        load_model_state(args.init_checkpoint, model)
        model.to(device)
        start_iter = 0
        print(f"加载 SFT 权重: {args.init_checkpoint}")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as exc:
            print(f"wandb 初始化失败, 改为本地训练: {exc}")

    print(f"DPO 开始: train_samples={len(train_dataset)}, valid_samples={len(valid_dataset)}, device={device}, batch_size={args.batch_size}, context={args.context_length}, beta={args.beta}")
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        chosen_x, chosen_y, chosen_mask, rejected_x, rejected_y, rejected_mask = get_dpo_batch(
            train_dataset, tokenizer, args.tokenizer_config, args.context_length, pad_id, args.batch_size, device
        )
        policy_chosen = masked_sequence_log_probs(model(chosen_x), chosen_y, chosen_mask)
        policy_rejected = masked_sequence_log_probs(model(rejected_x), rejected_y, rejected_mask)
        with torch.no_grad():
            reference_chosen = masked_sequence_log_probs(reference_model(chosen_x), chosen_y, chosen_mask)
            reference_rejected = masked_sequence_log_probs(reference_model(rejected_x), rejected_y, rejected_mask)
        loss, reward_margin = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, args.beta)

        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(), args.max_norm)
        optimizer.step()

        if it % args.log_interval == 0 or it == args.max_iters - 1:
            accuracy = (reward_margin > 0).float().mean().item()
            print(f"Iter {it}: dpo_loss {loss.item():.4f}, preference_accuracy {accuracy:.3f}, lr {lr:.6g}")
            if wandb_run is not None:
                wandb_run.log({"train/dpo_loss": loss.item(), "train/preference_accuracy": accuracy, "train/reward_margin": reward_margin.mean().item(), "lr": lr, "iter": it + 1})

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            val_loss, val_accuracy = evaluate(model, reference_model, valid_dataset, tokenizer, args.tokenizer_config, args.context_length, pad_id, args.batch_size, device, args.beta, args.eval_iters)
            print(f"Iter {it}: val_dpo_loss {val_loss:.4f}, val_preference_accuracy {val_accuracy:.3f}")
            if wandb_run is not None:
                wandb_run.log({"val/dpo_loss": val_loss, "val/preference_accuracy": val_accuracy, "iter": it + 1})

        if it % args.save_interval == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
