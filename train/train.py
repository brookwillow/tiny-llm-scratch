import torch
import math
import typing
import os
import sys
import numpy.typing as npt
from torch.optim import Optimizer
from collections.abc import Iterable
import argparse
import numpy as np
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model import Model, ModelConfig

def cross_entropy(logits, target):
    # logits:  (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)

    # 找到logits中的最大值，并保持形状为（batch_size, seq_len, 1）
    max_logits = torch.max(logits, dim = -1, keepdim=True).values

    # 根据target来取出logits中的对应的分值
    target_logits = torch.gather(logits, dim = -1, index = target.unsqueeze(-1)).squeeze(-1)

    # 减去max 值，保证运算安全
    shifted_logits = logits - max_logits

    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim = -1))

    loss = log_sum_exp - target_logits

    return torch.mean(loss)

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    计算第 it 次迭代时，带预热的余弦退火学习率。

    参数:
        it: 当前迭代步数 (t)
        max_learning_rate: 学习率的峰值 (alpha_max)
        min_learning_rate: 学习率的底值 (alpha_min)
        warmup_iters: 预热阶段的总步数 (T_w)
        cosine_cycle_iters: 整个衰减周期结束的步数 (T_c)
    """

    # 1. 预热阶段: 线性增长逻辑
    if it < warmup_iters:
        # 从 0 匀速增长到 max_learning_rate
        return max_learning_rate * it / warmup_iters

    # 2. 衰减周期后: 维持最小值
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦退火核心逻辑
    # a. 计算当前处于退火阶段的进度百分比 (0.0 到 1.0)
    # it - warmup_iters: 距离预热结束走了多少步
    # cosine_cycle_iters - warmup_iters: 整个退火阶段的总长度
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # b. 计算余弦系数
    # math.cos(math.pi * decay_ratio):
    # 当进度为 0 时, 结果为 cos(0) = 1
    # 当进度为 1 时, 结果为 cos(pi) = -1
    # coeff = 0.5 * (1 + [-1, 1]) -> 范围 [0.0, 1.0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # c. 最终计算
    # 学习率从 max 降向 min
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现全局梯度裁剪 (Global Norm Clipping)。

    参数:
        parameters: 模型的所有参数 (model.parameters())
        max_norm: 允许的最大梯度 L2 范数 (M)
    """
    # 1. 过滤掉没有梯度的参数 (防止对 None 对象操作)
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 2. 计算全局 L2 范数 (Global L2 Norm)
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 极其重要:
        # 梯度裁剪是在计算完导数后进行的数值操作，我们不希望“计算范数”的过程也被记入计算图。
        # torch.norm(..., p=2) 算出当前层梯度的 L2 范数 L_i
        param_norm = torch.norm(p.grad.detach(), p=2)

        # 将各层范数的平方累加 (L_total = sqrt(sum(L_i^2)))
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    # 3. 检查是否触发裁剪
    eps = 1e-6  # 防止除零的稳定性常数
    if total_norm > max_norm:
        # 计算统一的缩放系数
        clip_coef = max_norm / (total_norm + eps)

        # 4. 原地 (in-place) 修改每个参数的梯度
        # 使用 mul_ 直接修改内存，不产生临时副本，节省显存
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    max_seq_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样一个训练批次。

    返回:
        x: 输入张量, 形状 [batch_size, max_seq_length]
        y: 目标张量, 形状 [batch_size, max_seq_length]
    """
    n = len(dataset)
    # 最后一个可用的起点, 必须留出 max_seq_length 的空间给 x, 再多留 1 位给 y
    max_idx = n - max_seq_length - 1

    # 随机选择 batch_size 个起始点
    ix = torch.randint(0, max_idx + 1, (batch_size,))

    # 提取序列并转为 Numpy 数组, 再转为 Tensor
    # 这样做比循环里逐个 to(device) 快得多
    x = torch.stack([torch.from_numpy(dataset[i : i + max_seq_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1 : i + max_seq_length + 1].astype(np.int64)) for i in ix])

    # 一次性搬运到 GPU
    return x.to(device), y.to(device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    保存当前训练状态。
    """
    # 1. 构建一个包含所有必要信息的字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    # 2. 使用 torch.save 将字典写入目标 (可以是路径或文件流)
    torch.save(checkpoint, out)


def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点恢复状态, 并返回保存时的迭代次数。
    """
    # 1. 加载字典
    # 使用 map_location='cpu' 可以防止在没有 GPU 的机器上加载时报错
    checkpoint = torch.load(src, map_location='cpu')

    # 2. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 恢复优化器状态 (动量、步数等)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 4. 返回保存时的迭代次数
    return checkpoint['iteration']


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 基本参数检查
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 2. 将超参数存入 defaults 字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss = None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 3. 状态初始化（第一次运行步时执行）
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩（梯度的指数移动平均）
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩（梯度平方的指数移动平均）
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 4. 更新矩估计 (Algorithm 1)
                # m = beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. 计算偏差校正后的学习率 alpha_t
                # 这一步是为了消除初始值为 0 带来的偏移
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 6. 更新参数: theta = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                # 这是一个专门为优化器设计的复合算子，名字可以拆解为: add (加) + constant (常数)
                # p.addcdiv_(tensor1, tensor2, value=1.0)。 p=p+value*( tensor1 / tensor2 )
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 7. 应用解耦的权重衰减 (AdamW 的核心特性)
                # theta = theta - alpha_t * lambda * theta
                # p.add_(other, alpha=1.0)  p=p+(alpha*other)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def main():
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)

    # --- 实验/消融 (Ablation) 开关 ---
    # Ablation 1: 移除 RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm comp")
    # Ablation 2: Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre", "post"])
    # Ablation 3: 移除 ROPE (NOPE)
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional")
    # Ablation 4: SwiGLU vs SiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"])

    # --- 优化器超参数 ---
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)

    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB 设置 ---
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据 (使用 memmap)
    # 假设数据是以 uint16 存储的二进制文件
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    # np.memmap 延迟加载数据到内存, 非常适合大数据集, 并且将二进制文件转为 (uint16) 数组
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')

    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    # 2. 处理消融实验逻辑
    # 如果 no_rope 为 True, 则 theta 设为 None, TransformerBlock 内部就不会初始化 RoPE
    actual_rope_theta = None if args.no_rope else 10000.0
    # use_rms_norm 逻辑取反
    use_rms_norm = not args.no_rms_norm

    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
        device=args.device,
    )

    # 3. 初始化模型
    model = Model(config=config)
    model.to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}")

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        model.to(args.device)
        move_optimizer_state_to_device(optimizer, args.device)
        print(f"Resuming from iteration {start_iter}")

    # 6. 初始化 WandB 监控
    wandb.init(
        project="cs336-assignment1",
        name=args.run_name,
        config=args
    )

    # 7. 主训练循环
    for it in range(start_iter, args.max_iters):
        # A. 更新学习率
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)

        optimizer.step()

        # C. 验证与日志记录
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
            print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}")
            wandb.log({
                "train/loss": loss.item(),
                "val/loss": v_loss.item(),
                "lr": lr,
                "iter": it + 1
            })

        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    wandb.finish()

if __name__ == "__main__":
    main()
