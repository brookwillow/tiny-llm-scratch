"""
数据准备脚本：从 ModelScope 下载 pretrain_t2t.jsonl，使用 BPE tokenizer 编码后
保存为训练脚本可直接读取的 uint16 二进制文件。

用法:
    python data/prepare_data.py

输出:
    data/train.bin  — 训练集 (90%)
    data/val.bin    — 验证集 (10%)
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

from tokenizer.bpe_tokenizer import BPETokenizer

# --- 配置 ---
DATASET_ID = "gongjy/minimind_dataset"
DATASET_FILE = "pretrain_t2t.jsonl"
DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / DATASET_FILE
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
VAL_RATIO = 0.1
_WORKER_TOKENIZER = None


def download_dataset():
    """从 ModelScope 下载数据集文件（如果本地不存在）"""
    if RAW_FILE.exists():
        print(f"数据集已存在: {RAW_FILE}")
        return

    print(f"正在从 ModelScope 下载 {DATASET_ID}/{DATASET_FILE} ...")
    try:
        from modelscope import snapshot_download
    except ImportError:
        from modelscope.hub.snapshot_download import snapshot_download

    last_error = None
    download_kwargs_list = [
        {
            "repo_type": "dataset",
            "allow_file_pattern": DATASET_FILE,
            "local_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_patterns": DATASET_FILE,
            "local_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_file_pattern": DATASET_FILE,
            "cache_dir": str(DATA_DIR),
        },
        {
            "repo_type": "dataset",
            "allow_patterns": DATASET_FILE,
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

    if not RAW_FILE.exists():
        matched_files = list(downloaded_dir.rglob(DATASET_FILE))
        if matched_files:
            RAW_FILE.write_bytes(matched_files[0].read_bytes())

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"下载完成但未找到 {RAW_FILE}，请检查 ModelScope 缓存目录: {downloaded_dir}")

    print(f"下载完成: {RAW_FILE}")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pretrain token bins from JSONL text data.")
    parser.add_argument("--raw_file", type=str, default=str(RAW_FILE))
    parser.add_argument("--train_bin", type=str, default=str(TRAIN_BIN))
    parser.add_argument("--val_bin", type=str, default=str(VAL_BIN))
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
        download_dataset()
    tokenize_and_save(args)
