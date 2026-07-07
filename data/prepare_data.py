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
import numpy as np
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


def tokenize_and_save():
    """将 JSONL 数据编码为 token ids 并保存为二进制文件"""
    if TRAIN_BIN.exists() and VAL_BIN.exists():
        print(f"二进制文件已存在: {TRAIN_BIN}, {VAL_BIN}")
        print("如需重新生成，请删除后重新运行。")
        return

    # 初始化 tokenizer
    vocab_path = ROOT_DIR / "tokenizer" / "vocab.json"
    merges_path = ROOT_DIR / "tokenizer" / "merges.txt"
    tokenizer = BPETokenizer(str(vocab_path), str(merges_path))

    # 读取 JSONL 并编码
    print(f"正在读取并编码 {RAW_FILE} ...")
    all_tokens = []
    line_count = 0

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # pretrain_t2t.jsonl 格式: {"text": "..."} 
            text = obj.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  已处理 {line_count} 行, 累计 {len(all_tokens)} tokens")

    print(f"编码完成: {line_count} 行, 共 {len(all_tokens)} tokens")

    # 转为 numpy 数组
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # 划分训练集和验证集
    n = len(all_tokens)
    val_size = int(n * VAL_RATIO)
    train_size = n - val_size

    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]

    # 保存为二进制文件
    train_tokens.tofile(TRAIN_BIN)
    val_tokens.tofile(VAL_BIN)

    print(f"训练集: {train_size} tokens -> {TRAIN_BIN}")
    print(f"验证集: {val_size} tokens -> {VAL_BIN}")


if __name__ == "__main__":
    download_dataset()
    tokenize_and_save()
