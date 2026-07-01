# 从零训练一个 LLM

> 本项目记录从零开始构建并训练一个小型语言模型（LLM）的完整过程，包括每一个关键步骤的代码、思考与实验记录。

## 数据集

### 来源

使用 [MiniMind](https://github.com/jingyaogong/minimind) 项目配套数据集：

- **ModelScope**：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- **HuggingFace**：https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main

### 文件列表


| 文件名                    | 大小   | 用途                          |
| ------------------------- | ------ | ----------------------------- |
| `pretrain_t2t_mini.jsonl` | 1.2 GB | 轻量预训练（快速复现）        |
| `pretrain_t2t.jsonl`      | 10 GB  | 完整预训练                    |
| `sft_t2t_mini.jsonl`      | 1.6 GB | 轻量 SFT 微调（含 Tool Call） |
| `sft_t2t.jsonl`           | 14 GB  | 完整 SFT                      |
| `rlaif.jsonl`             | 24 MB  | PPO / GRPO / CISPO 强化学习   |
| `dpo.jsonl`               | 53 MB  | RLHF 偏好对齐训练             |
| `agent_rl.jsonl`          | 86 MB  | 多轮 Tool-Use 强化学习        |
| `agent_rl_math.jsonl`     | 18 MB  | 数学推理 / 工具 RL            |

### 数据格式

**预训练数据**（`pretrain_*.jsonl`）— 纯文本续写：

```json
{"text": "Transformer 通过自注意力机制建模上下文关系..."}
```

**SFT 对话数据**（`sft_*.jsonl`）— 多轮对话格式：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
  ]
}
```

**SFT 工具调用数据**（已混入主线 SFT）— Tool Call 格式：

```json
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",...}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

**DPO 偏好数据**（`dpo.jsonl`）— chosen / rejected 对：

```json
{
  "chosen":   [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "好回答"}],
  "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "差回答"}]
}
```

**RLAIF 数据**（`rlaif.jsonl`）— assistant 留空，由策略模型在线生成：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释一下什么是光合作用？"},
    {"role": "assistant", "content": "无"}
  ]
}
```

## 项目结构

```
tiny-llm-scratch/
├── README.md                    # 本文件，进度总览
├── dataset/                     # 训练数据（从 ModelScope 下载后放这里）
│   ├── pretrain_t2t_mini.jsonl
│   └── sft_t2t_mini.jsonl
├── tokenizer/                   # BPE Tokenizer 训练
│   ├── README.md                # 训练方案说明
│   ├── train.py                 # 训练脚本（待实现）
│   ├── test.py                  # 验证脚本（待实现）
│   └── output/                  # 训练产物（不提交 git）
│       ├── tokenizer.json
│       └── vocab.txt
├── model/                       # 模型定义（待实现）
├── train.py                     # 预训练入口（待实现）
├── generate.py                  # 推理入口（待实现）
└── configs/                     # 超参数配置（待实现）
```

## 超参数预设（Tiny 版本）


| 参数             | 值   | 说明             |
| ---------------- | ---- | ---------------- |
| `vocab_size`     | 待定 | 取决于 Tokenizer |
| `context_length` | 256  | 最大上下文长度   |
| `d_model`        | 128  | 词向量维度       |
| `n_heads`        | 4    | 注意力头数       |
| `n_layers`       | 4    | Transformer 层数 |
| `d_ff`           | 512  | FFN 隐层维度     |
| `dropout`        | 0.1  | Dropout 比率     |
| `batch_size`     | 64   | 每批样本数       |
| `lr`             | 3e-4 | 初始学习率       |

## 环境依赖（待补充）

```bash
Python >= 3.10
torch >= 2.0
```

## 参考资料

- [MiniMind](https://github.com/jingyaogong/minimind) — 本项目数据集与参考实现来源
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [GPT-2 论文](https://openai.com/research/language-unsupervised) — Decoder-Only 架构参考
- [Andrej Karpathy - nanoGPT](https://github.com/karpathy/nanoGPT) — 极简 GPT 实现参考
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — 逐行注释版 Transformer

## 日志


| 日期       | 完成内容                                   |
| ---------- | ------------------------------------------ |
| 2026-07-01 | 项目初始化，创建 README                    |
| 2026-07-01 | 确定数据集：MiniMind Dataset（ModelScope） |
