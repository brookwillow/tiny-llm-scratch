"""
数据准备脚本：从 ModelScope 下载 pretrain_t2t.jsonl / sft_t2t.jsonl，并完成训练前处理。

pretrain 数据会编码为训练脚本可直接读取的 uint16 二进制文件。
SFT 数据只做格式归一化和 train/val 拆分，不在 prepare 阶段 tokenizer。

用法:
    python data/prepare_data.py --mode pretrain
    python data/prepare_data.py --mode sft
    python data/prepare_data.py --mode all

输出:
    data/train.bin  — 训练集 (90%)
    data/val.bin    — 验证集 (10%)
    data/sft_train.jsonl — SFT 训练集 raw messages
    data/sft_val.jsonl   — SFT 验证集 raw messages
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from array import array
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config.model_config import add_model_config_arg
from tokenizer.bpe_tokenizer import BPETokenizer

# --- 配置 ---
DATASET_ID = "gongjy/minimind_dataset"
DATASET_FILE = "pretrain_t2t.jsonl"
SFT_DATASET_FILE = "sft_t2t.jsonl"
DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / DATASET_FILE
SFT_RAW_FILE = DATA_DIR / SFT_DATASET_FILE
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
SFT_TRAIN_JSONL = DATA_DIR / "sft_train.jsonl"
SFT_VAL_JSONL = DATA_DIR / "sft_val.jsonl"
VAL_RATIO = 0.1
_WORKER_TOKENIZER = None


def download_dataset_file(dataset_file: str, output_file: Path):
    """从 ModelScope 下载数据集文件（如果本地不存在）"""
    if output_file.exists():
        print(f"数据集已存在: {output_file}")
        return

    print(f"正在从 ModelScope 下载 {DATASET_ID}/{dataset_file} ...")
    try:
        from modelscope import snapshot_download
    except ImportError:
        from modelscope.hub.snapshot_download import snapshot_download

    last_error = None
    download_kwargs_list = [
        {
            "repo_type": "dataset",
            "allow_file_pattern": dataset_file,
            "local_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_patterns": dataset_file,
            "local_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_file_pattern": dataset_file,
            "cache_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_patterns": dataset_file,
            "cache_dir": str(DATA_DIR),
        },
    ]

    for kwargs in download_kwargs_list:
        try:
            downloaded_dir = Path(snapshot_download(DATASET_ID, **kwargs))
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise RuntimeError("当前 modelscope 版本不支持脚本中的 snapshot_download 参数") from last_error

    if not output_file.exists():
        matched_files = list(downloaded_dir.rglob(dataset_file))
        if matched_files:
            output_file.write_bytes(matched_files[0].read_bytes())

    if not output_file.exists():
        raise FileNotFoundError(f"下载完成但未找到 {output_file}，请检查 ModelScope 缓存目录: {downloaded_dir}")

    print(f"下载完成: {output_file}")


def download_dataset():
    download_dataset_file(DATASET_FILE, RAW_FILE)


def download_sft_dataset():
    download_dataset_file(SFT_DATASET_FILE, SFT_RAW_FILE)


def init_worker(vocab_path: str, merges_path: str):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer(vocab_path, merges_path)


def encode_text_worker(text: str) -> list[int]:
    return _WORKER_TOKENIZER.encode(text)


def iter_jsonl_texts(raw_file: Path, max_lines: int | None = None):
    emitted = 0
    with raw_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if max_lines is not None and emitted >= max_lines:
                break

            line = raw_line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue

            emitted += 1
            yield text


def count_jsonl_lines(raw_file: Path, max_lines: int | None = None) -> int:
    count = 0
    with raw_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if max_lines is not None and count >= max_lines:
                break
            if raw_line.strip():
                count += 1
    return count


def write_uint16_tokens(file_obj, tokens: list[int]) -> None:
    if not tokens:
        return
    array("H", tokens).tofile(file_obj)


def format_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def should_write_val(processed_lines: int, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True
    val_period = max(1, round(1 / val_ratio))
    return processed_lines % val_period == 0


def print_progress(
    processed_lines: int,
    train_tokens: int,
    val_tokens: int,
    started_at: float,
    total_lines: int | None,
) -> None:
    elapsed = time.time() - started_at
    lines_per_sec = processed_lines / elapsed if elapsed > 0 else 0.0
    tokens = train_tokens + val_tokens
    tokens_per_sec = tokens / elapsed if elapsed > 0 else 0.0
    if total_lines and lines_per_sec > 0:
        remaining_lines = max(0, total_lines - processed_lines)
        eta = remaining_lines / lines_per_sec
        percent = processed_lines / total_lines * 100
        progress = f"{processed_lines:,}/{total_lines:,} ({percent:.2f}%)"
    else:
        eta = float("inf")
        progress = f"{processed_lines:,}"

    print(
        "  "
        f"已处理 {progress} 行, "
        f"tokens={tokens:,} "
        f"(train={train_tokens:,}, val={val_tokens:,}), "
        f"{lines_per_sec:.1f} lines/s, {tokens_per_sec:.1f} tokens/s, "
        f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)}",
        flush=True,
    )


def tokenize_jsonl_to_bins(
    raw_file: Path,
    train_bin: Path,
    val_bin: Path,
    tokenizer: BPETokenizer | None,
    val_ratio: float,
    max_lines: int | None,
    num_workers: int,
    chunk_size: int,
    log_interval: int,
    total_lines: int | None,
    vocab_path: Path | None = None,
    merges_path: Path | None = None,
) -> dict[str, int]:
    processed_lines = 0
    train_tokens = 0
    val_tokens = 0
    started_at = time.time()

    train_bin.parent.mkdir(parents=True, exist_ok=True)
    val_bin.parent.mkdir(parents=True, exist_ok=True)

    text_iter = iter_jsonl_texts(raw_file, max_lines=max_lines)

    with train_bin.open("wb") as train_f, val_bin.open("wb") as val_f:
        if num_workers <= 1:
            if tokenizer is None:
                if vocab_path is None or merges_path is None:
                    raise ValueError("tokenizer or vocab_path/merges_path is required")
                tokenizer = BPETokenizer(str(vocab_path), str(merges_path))
            encoded_iter = (tokenizer.encode(text) for text in text_iter)
        else:
            if vocab_path is None or merges_path is None:
                raise ValueError("vocab_path and merges_path are required for multiprocessing")
            pool = mp.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(str(vocab_path), str(merges_path)),
            )
            encoded_iter = pool.imap(encode_text_worker, text_iter, chunksize=chunk_size)

        try:
            for tokens in encoded_iter:
                processed_lines += 1
                if should_write_val(processed_lines, val_ratio):
                    write_uint16_tokens(val_f, tokens)
                    val_tokens += len(tokens)
                else:
                    write_uint16_tokens(train_f, tokens)
                    train_tokens += len(tokens)

                if log_interval > 0 and processed_lines % log_interval == 0:
                    print_progress(processed_lines, train_tokens, val_tokens, started_at, total_lines)
        finally:
            if num_workers > 1:
                pool.close()
                pool.join()

    print_progress(processed_lines, train_tokens, val_tokens, started_at, total_lines)
    return {
        "processed_lines": processed_lines,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }


def print_sft_progress(
    processed_lines: int,
    train_samples: int,
    val_samples: int,
    started_at: float,
    total_lines: int | None,
) -> None:
    elapsed = time.time() - started_at
    lines_per_sec = processed_lines / elapsed if elapsed > 0 else 0.0
    if total_lines and lines_per_sec > 0:
        remaining_lines = max(0, total_lines - processed_lines)
        eta = remaining_lines / lines_per_sec
        percent = processed_lines / total_lines * 100
        progress = f"{processed_lines:,}/{total_lines:,} ({percent:.2f}%)"
    else:
        eta = float("inf")
        progress = f"{processed_lines:,}"

    print(
        "  "
        f"已处理 {progress} 行, "
        f"samples={train_samples + val_samples:,} "
        f"(train={train_samples:,}, val={val_samples:,}), "
        f"{lines_per_sec:.1f} lines/s, "
        f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)}",
        flush=True,
    )


def split_sft_jsonl_to_jsonl(
    raw_file: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    val_ratio: float,
    max_lines: int | None,
    log_interval: int,
    total_lines: int | None,
) -> dict[str, int]:
    from train.train_sft import normalize_sft_messages

    processed_lines = 0
    train_samples = 0
    val_samples = 0
    skipped_lines = 0
    started_at = time.time()

    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with raw_file.open("r", encoding="utf-8") as raw_f, train_jsonl.open("w", encoding="utf-8") as train_f, val_jsonl.open(
        "w", encoding="utf-8"
    ) as val_f:
        for line_no, raw_line in enumerate(raw_f, start=1):
            if max_lines is not None and processed_lines >= max_lines:
                break
            if not raw_line.strip():
                continue

            try:
                sample = json.loads(raw_line)
                messages = normalize_sft_messages(sample)
                if not any(message["role"] == "assistant" for message in messages):
                    skipped_lines += 1
                    continue

                processed_lines += 1
                out_line = json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
                if should_write_val(processed_lines, val_ratio):
                    val_f.write(out_line)
                    val_samples += 1
                else:
                    train_f.write(out_line)
                    train_samples += 1
            except Exception as exc:
                skipped_lines += 1
                if skipped_lines <= 5:
                    print(f"跳过第 {line_no} 行: {exc}")

            if log_interval > 0 and processed_lines > 0 and processed_lines % log_interval == 0:
                print_sft_progress(processed_lines, train_samples, val_samples, started_at, total_lines)

    print_sft_progress(processed_lines, train_samples, val_samples, started_at, total_lines)
    return {
        "processed_lines": processed_lines,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "skipped_lines": skipped_lines,
    }


def tokenize_and_save(args):
    """将 JSONL 数据编码为 token ids 并流式保存为二进制文件"""
    train_bin = Path(args.train_bin)
    val_bin = Path(args.val_bin)
    raw_file = Path(args.raw_file)

    if (train_bin.exists() or val_bin.exists()) and not args.overwrite:
        print(f"输出文件已存在: {train_bin}, {val_bin}")
        print("如需重新生成，请加 --overwrite。")
        return

    vocab_path = ROOT_DIR / "tokenizer" / "vocab.json"
    merges_path = ROOT_DIR / "tokenizer" / "merges.txt"

    total_lines = None
    if not args.no_count_total:
        print("正在统计总行数用于 ETA ...")
        total_lines = count_jsonl_lines(raw_file, max_lines=args.max_lines)
        print(f"预计处理行数: {total_lines:,}")

    tokenizer = None
    if args.num_workers <= 1:
        tokenizer = BPETokenizer(str(vocab_path), str(merges_path))

    print(f"正在读取并编码 {raw_file} ...")
    print(
        f"num_workers={args.num_workers}, chunk_size={args.chunk_size}, "
        f"max_lines={args.max_lines}, val_ratio={args.val_ratio}"
    )
    stats = tokenize_jsonl_to_bins(
        raw_file=raw_file,
        train_bin=train_bin,
        val_bin=val_bin,
        tokenizer=tokenizer,
        val_ratio=args.val_ratio,
        max_lines=args.max_lines,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        log_interval=args.log_interval,
        total_lines=total_lines,
        vocab_path=vocab_path,
        merges_path=merges_path,
    )

    print("编码完成")
    print(f"处理行数: {stats['processed_lines']:,}")
    print(f"训练集: {stats['train_tokens']:,} tokens -> {train_bin}")
    print(f"验证集: {stats['val_tokens']:,} tokens -> {val_bin}")


def split_sft_and_save(args):
    """将 SFT JSONL 数据归一化为 raw messages，并拆分 train/val"""
    raw_file = Path(args.sft_raw_file)
    train_jsonl = Path(args.sft_train_jsonl)
    val_jsonl = Path(args.sft_val_jsonl)

    if (train_jsonl.exists() or val_jsonl.exists()) and not args.overwrite:
        print(f"输出文件已存在: {train_jsonl}, {val_jsonl}")
        print("如需重新生成，请加 --overwrite。")
        return

    total_lines = None
    if not args.no_count_total:
        print("正在统计 SFT 总行数用于 ETA ...")
        total_lines = count_jsonl_lines(raw_file, max_lines=args.max_lines)
        print(f"预计处理 SFT 行数: {total_lines:,}")

    print(f"正在读取并拆分 SFT 数据 {raw_file} ...")
    print(f"max_lines={args.max_lines}, val_ratio={args.val_ratio}")
    stats = split_sft_jsonl_to_jsonl(
        raw_file=raw_file,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        val_ratio=args.val_ratio,
        max_lines=args.max_lines,
        log_interval=args.log_interval,
        total_lines=total_lines,
    )

    print("SFT 拆分完成")
    print(f"处理行数: {stats['processed_lines']:,}")
    print(f"跳过行数: {stats['skipped_lines']:,}")
    print(f"SFT 训练集: {stats['train_samples']:,} samples -> {train_jsonl}")
    print(f"SFT 验证集: {stats['val_samples']:,} samples -> {val_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pretrain token bins and raw SFT JSONL splits.")
    add_model_config_arg(parser, default=str(ROOT_DIR / "configs" / "model_512x6.json"))
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft", "all"])
    parser.add_argument("--raw_file", type=str, default=str(RAW_FILE))
    parser.add_argument("--train_bin", type=str, default=str(TRAIN_BIN))
    parser.add_argument("--val_bin", type=str, default=str(VAL_BIN))
    parser.add_argument("--sft_raw_file", type=str, default=str(SFT_RAW_FILE))
    parser.add_argument("--sft_train_jsonl", type=str, default=str(SFT_TRAIN_JSONL))
    parser.add_argument("--sft_val_jsonl", type=str, default=str(SFT_VAL_JSONL))
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--no_count_total", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.no_download:
        if args.mode in {"pretrain", "all"}:
            download_dataset()
        if args.mode in {"sft", "all"}:
            download_sft_dataset()
    if args.mode in {"pretrain", "all"}:
        tokenize_and_save(args)
    if args.mode in {"sft", "all"}:
        split_sft_and_save(args)
