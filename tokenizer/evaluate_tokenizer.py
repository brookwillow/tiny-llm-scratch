from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bpe_tokenizer import BPETokenizer


@dataclass(frozen=True)
class EvalSample:
    category: str
    text: str


DEFAULT_SAMPLES = [
    EvalSample("zh_general", "你好，世界！这是一个用于评估分词器效果的中文句子。"),
    EvalSample("zh_general", "我们希望 tokenizer 能够稳定处理中文、英文、数字和标点符号。"),
    EvalSample("zh_dialog", "用户：帮我写一个排序函数。\n助手：可以，请告诉我输入和输出格式。"),
    EvalSample("zh_technical", "大模型训练通常包括数据清洗、分词、预训练、微调和评估几个阶段。"),
    EvalSample("en_general", "Hello, world! This is a compact tokenizer evaluation sentence."),
    EvalSample("en_general", "The quick brown fox jumps over the lazy dog near the river bank."),
    EvalSample("code_python", "def merge_lists(a, b):\n    return sorted(set(a + b))\n"),
    EvalSample("code_python", "for i in range(10):\n    print(f'value={i}')\n"),
    EvalSample("mixed", "今天的 temperature is 28.5°C, humidity is 72%, 风有点大。"),
    EvalSample("mixed", "请把 JSON 字符串 {\"name\": \"Alice\", \"age\": 18} 解析出来。"),
    EvalSample("numbers", "订单号 202607020001，金额 ¥1,234.56，折扣 15%。"),
    EvalSample("punctuation", "！？。，；：“”‘’（）[]{}<>/\\|~`@#$%^&*-_+="),
    EvalSample("emoji", "模型表现不错 👍🚀，但还需要处理 emoji、组合符号和罕见字符。"),
    EvalSample("multilingual", "こんにちは世界。안녕하세요 세계. Привет, мир. مرحبا بالعالم."),
    EvalSample("whitespace", "第一行\n\n第二行\t带有 tab    和多个空格。\n"),
    EvalSample("domain_unseen", "CRISPR-Cas9 基因编辑、量子纠缠和拓扑绝缘体都是专业领域词汇。"),
]


def is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_samples(tokenizer: Any, samples: list[EvalSample]) -> dict[str, Any]:
    rows = []
    category_totals: dict[str, dict[str, Any]] = {}

    for sample in samples:
        token_ids = tokenizer.encode(sample.text)
        decoded = tokenizer.decode(token_ids)
        char_count = len(sample.text)
        byte_count = len(sample.text.encode("utf-8"))
        token_count = len(token_ids)
        cjk_count = sum(1 for ch in sample.text if is_cjk_char(ch))
        ok = decoded == sample.text

        row = {
            "category": sample.category,
            "text": sample.text,
            "roundtrip_ok": ok,
            "chars": char_count,
            "bytes": byte_count,
            "tokens": token_count,
            "cjk_chars": cjk_count,
            "tokens_per_char": safe_div(token_count, char_count),
            "bytes_per_token": safe_div(byte_count, token_count),
            "chars_per_token": safe_div(char_count, token_count),
        }
        rows.append(row)

        bucket = category_totals.setdefault(
            sample.category,
            {
                "samples": 0,
                "roundtrip_errors": 0,
                "chars": 0,
                "bytes": 0,
                "tokens": 0,
                "cjk_chars": 0,
            },
        )
        bucket["samples"] += 1
        bucket["roundtrip_errors"] += 0 if ok else 1
        bucket["chars"] += char_count
        bucket["bytes"] += byte_count
        bucket["tokens"] += token_count
        bucket["cjk_chars"] += cjk_count

    total = {
        "samples": len(rows),
        "roundtrip_errors": sum(0 if row["roundtrip_ok"] else 1 for row in rows),
        "chars": sum(row["chars"] for row in rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
        "cjk_chars": sum(row["cjk_chars"] for row in rows),
    }

    enrich_summary(total, rows)
    for bucket in category_totals.values():
        enrich_summary(bucket, None)

    return {
        "total": total,
        "by_category": dict(sorted(category_totals.items())),
        "samples": rows,
    }


def enrich_summary(summary: dict[str, Any], rows: list[dict[str, Any]] | None) -> None:
    summary["tokens_per_char"] = safe_div(summary["tokens"], summary["chars"])
    summary["bytes_per_token"] = safe_div(summary["bytes"], summary["tokens"])
    summary["chars_per_token"] = safe_div(summary["chars"], summary["tokens"])
    summary["tokens_per_cjk_char"] = safe_div(summary["tokens"], summary["cjk_chars"])
    if rows:
        token_counts = [row["tokens"] for row in rows]
        summary["min_tokens"] = min(token_counts)
        summary["max_tokens"] = max(token_counts)
        summary["mean_tokens"] = statistics.mean(token_counts)


def print_report(report: dict[str, Any], show_samples: bool) -> None:
    total = report["total"]
    print("Tokenizer evaluation")
    print("=" * 80)
    print(
        "total: "
        f"samples={total['samples']} "
        f"roundtrip_errors={total['roundtrip_errors']} "
        f"chars={total['chars']} "
        f"bytes={total['bytes']} "
        f"tokens={total['tokens']} "
        f"tokens/char={total['tokens_per_char']:.3f} "
        f"bytes/token={total['bytes_per_token']:.3f}"
    )
    print("\nBy category")
    print("-" * 80)
    print(f"{'category':<16} {'n':>3} {'err':>3} {'chars':>7} {'bytes':>7} {'tokens':>7} {'tok/char':>8} {'byte/tok':>8}")
    for category, row in report["by_category"].items():
        print(
            f"{category:<16} {row['samples']:>3} {row['roundtrip_errors']:>3} "
            f"{row['chars']:>7} {row['bytes']:>7} {row['tokens']:>7} "
            f"{row['tokens_per_char']:>8.3f} {row['bytes_per_token']:>8.3f}"
        )

    if not show_samples:
        return

    print("\nSamples")
    print("-" * 80)
    for row in report["samples"]:
        status = "OK" if row["roundtrip_ok"] else "FAIL"
        preview = row["text"].replace("\n", "\\n")
        if len(preview) > 56:
            preview = preview[:53] + "..."
        print(
            f"[{status}] {row['category']:<16} "
            f"tokens={row['tokens']:>4} "
            f"tok/char={row['tokens_per_char']:.3f} "
            f"byte/tok={row['bytes_per_token']:.3f}  {preview}"
        )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate tokenizer compression and round-trip quality.")
    parser.add_argument("--vocab", default=str(base_dir / "vocab.json"))
    parser.add_argument("--merges", default=str(base_dir / "merges.txt"))
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print full report as JSON.")
    parser.add_argument("--samples", action="store_true", help="Print per-sample metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = BPETokenizer(args.vocab, args.merges)
    report = summarize_samples(tokenizer, DEFAULT_SAMPLES)
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, show_samples=args.samples)


if __name__ == "__main__":
    main()
