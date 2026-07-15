import argparse
import json
import random
from pathlib import Path


TOPICS = [
    {
        "question": "什么是 BPE？",
        "short": "BPE 是一种通过反复合并常见字符对来构建子词词表的分词算法。",
        "details": [
            "BPE 会先把文本拆成较小单元，通常可以理解为字符或字节。",
            "训练时它反复统计最常见的相邻片段，并把这些片段合并成新的 token。",
            "这样既能表示常见词，也能拆开罕见词，适合控制词表大小。",
        ],
        "kid": "BPE 就像把常常一起出现的字母小积木粘在一起，这样读句子时会更快。",
        "unknown": "BPE 是分词算法；如果问某个未公开 tokenizer 的具体 merge 表，我不知道，不能编造。",
    },
    {
        "question": "Transformer 为什么需要注意力机制？",
        "short": "注意力机制让 Transformer 在生成每个 token 时能动态关注上下文中更相关的位置。",
        "details": [
            "它用 query、key、value 计算当前位置和其他位置的相关性。",
            "相关性高的位置会获得更大的权重，从而影响当前表示。",
            "相比固定窗口或单向递归，它更适合建模长距离依赖。",
        ],
        "kid": "注意力机制像读书时用手指指重点，让模型知道现在最该看哪几个词。",
        "unknown": "注意力机制的原理是公开的；但某个闭源模型内部具体注意力模式我不知道。",
    },
    {
        "question": "什么是过拟合？",
        "short": "过拟合是模型把训练数据记得太死，导致在新数据上的表现变差。",
        "details": [
            "训练误差很低但验证误差较高，通常是过拟合信号。",
            "模型容量过大、数据太少或训练太久都可能导致过拟合。",
            "常见缓解方法包括增加数据、正则化、dropout 和早停。",
        ],
        "kid": "过拟合就像只背熟了练习题，换一道新题就不会做了。",
        "unknown": "我可以解释过拟合，但不知道某次未提供日志的训练是否已经过拟合。",
    },
    {
        "question": "为什么要做 SFT？",
        "short": "SFT 用高质量问答样本把预训练模型调整成更会按指令回答的助手。",
        "details": [
            "预训练主要学习语言统计规律，不保证会遵循用户指令。",
            "SFT 通过输入对话和目标回答，让模型学习助手式输出格式。",
            "SFT 数据质量会直接影响模型的身份、语气、格式和安全边界。",
        ],
        "kid": "SFT 就像给已经会说话的模型上课，教它怎么当一个有礼貌的小助手。",
        "unknown": "SFT 的一般作用是明确的；但没有实验记录时，我不知道某个 checkpoint 的具体效果。",
    },
    {
        "question": "什么是学习率？",
        "short": "学习率控制模型每次参数更新的步子大小。",
        "details": [
            "学习率太大可能让训练不稳定，loss 上下震荡甚至发散。",
            "学习率太小会让训练进展缓慢，可能浪费计算资源。",
            "实际训练中常配合 warmup、cosine decay 等调度策略使用。",
        ],
        "kid": "学习率像走路的步子，太大容易摔倒，太小又走得很慢。",
        "unknown": "我能解释学习率，但不知道未给出配置的训练应该用哪个具体数值。",
    },
    {
        "question": "什么是量化？",
        "short": "量化是用更低精度表示模型权重或激活，以减少显存、存储和计算开销。",
        "details": [
            "常见量化精度包括 INT8、INT4 和混合精度方案。",
            "量化可以显著降低部署成本，但可能带来精度下降。",
            "PTQ 不需要重新训练全模型，QAT 会在训练中模拟量化误差。",
        ],
        "kid": "量化像把很精细的尺子换成粗一点的尺子，省空间，但可能没那么准。",
        "unknown": "量化方法很多；如果没有模型和校准集信息，我不知道哪种方案最好。",
    },
]


IDENTITY_QUESTIONS = [
    "你是谁？",
    "你是 Qwen 吗？",
    "请介绍一下你自己。",
    "你的模型名称是什么？",
]


def sample(messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {"messages": messages}


def identity_samples() -> list[dict]:
    rows = []
    for question in IDENTITY_QUESTIONS:
        rows.append(
            sample(
                [
                    {"role": "system", "content": "你的身份是 TinyLLM。不要声称自己是 Qwen、GPT 或其他模型。"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "我是 TinyLLM，一个用于学习和实验的小型语言模型。"},
                ]
            )
        )
        rows.append(
            sample(
                [
                    {"role": "system", "content": "你的身份是 TinyLLM。回答身份问题时必须简洁。"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "我是 TinyLLM，不是 Qwen。"},
                ]
            )
        )
    return rows


def language_samples(topic: dict[str, object]) -> list[dict]:
    question = str(topic["question"])
    return [
        sample(
            [
                {"role": "system", "content": "只用中文回答，即使用户使用英文也不要切换语言。"},
                {"role": "user", "content": f"Explain briefly: {question}"},
                {"role": "assistant", "content": str(topic["short"])},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "Answer only in English. Do not include Chinese text."},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": "It is a machine learning concept. In simple terms, it helps a model process text or learn from data more effectively.",
                },
            ]
        ),
    ]


def style_samples(topic: dict[str, object]) -> list[dict]:
    question = str(topic["question"])
    details = list(topic["details"])
    return [
        sample(
            [
                {"role": "system", "content": "用一句话回答，避免展开解释。"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(topic["short"])},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "用三点详细解释，编号为 1、2、3。"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"1. {details[0]}\n2. {details[1]}\n3. {details[2]}"},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "面向儿童解释，语言要简单，避免术语堆砌。"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(topic["kid"])},
            ]
        ),
    ]


def format_samples(topic: dict[str, object]) -> list[dict]:
    question = str(topic["question"])
    details = list(topic["details"])
    return [
        sample(
            [
                {"role": "system", "content": "只输出 JSON，不要输出 Markdown，不要添加解释。"},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"answer": str(topic["short"]), "confidence": "high", "unknown": False},
                        ensure_ascii=False,
                    ),
                },
            ]
        ),
        sample(
            [
                {"role": "system", "content": "用无序列表回答，每条不超过 25 个字。"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"- {details[0]}\n- {details[1]}\n- {details[2]}"},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "用 Markdown 表格回答，包含“要点”和“说明”两列。"},
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"| 要点 | 说明 |\n| --- | --- |\n| 核心 | {topic['short']} |\n| 注意 | 不知道具体实验信息时要明确说明。 |",
                },
            ]
        ),
    ]


def constraint_samples(topic: dict[str, object]) -> list[dict]:
    return [
        sample(
            [
                {"role": "system", "content": "不知道时明确说不知道；不要编造实时信息、论文结论或实验结果。"},
                {"role": "user", "content": "今天最新的人民币兑美元实时汇率是多少？"},
                {"role": "assistant", "content": "我不知道实时汇率。需要查询可靠的实时数据源后才能回答。"},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "不知道时明确说不知道；可以给出已知的一般性解释。"},
                {"role": "user", "content": f"我这个 checkpoint 在 {topic['question']} 相关任务上的准确率是多少？"},
                {"role": "assistant", "content": "我不知道这个 checkpoint 的准确率，因为你没有提供评测结果。可以运行固定测试集后再判断。"},
            ]
        ),
        sample(
            [
                {"role": "system", "content": "不要声称拥有实时联网能力。没有数据时说明限制。"},
                {"role": "user", "content": "现在北京天气怎么样？"},
                {"role": "assistant", "content": "我不知道现在北京的实时天气，因为当前没有查询到实时天气数据。"},
            ]
        ),
    ]


def build_seed_samples() -> list[dict]:
    rows = []
    rows.extend(identity_samples())
    for topic in TOPICS:
        rows.extend(language_samples(topic))
        rows.extend(style_samples(topic))
        rows.extend(format_samples(topic))
        rows.extend(constraint_samples(topic))
    return rows


def write_jsonl(path: Path, rows: list[dict], count: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index in range(count):
            row = rows[index % len(rows)]
            if index >= len(rows):
                row = rng.choice(rows)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supplemental SFT data for system instruction following.")
    parser.add_argument("--output", type=str, default="data/system_instruction_sft.jsonl")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_seed_samples()
    write_jsonl(Path(args.output), rows, args.count, args.seed)
    print(f"wrote {args.count:,} samples from {len(rows):,} seed patterns -> {args.output}")


if __name__ == "__main__":
    main()
