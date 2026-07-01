import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 标准做法：建立 256 个字节 → 唯一可打印 Unicode 字符的映射表。

    规则：
      - 已经可打印的字节（ASCII 可打印 + Latin 可打印）直接映射为对应字符
      - 其余 68 个字节（控制字符、不可见字符）映射到 Unicode 256 之后的字符
        Ā ā Ă ă Ą ą ...（不会和任何正常字符冲突）

    这样每个字节都有唯一确定的可打印字符表示，vocab.json 里不会出现乱码或空方块。
    """
    printable = (
        list(range(ord('!'),  ord('~')  + 1)) +  # 33~126：可打印 ASCII
        list(range(ord('¡'),  ord('¬')  + 1)) +  # 161~172：Latin 可打印
        list(range(ord('®'),  ord('ÿ')  + 1))    # 174~255：Latin 可打印
    )
    b2u = {b: chr(b) for b in printable}

    extra_code = 256
    for b in range(256):
        if b not in b2u:
            b2u[b] = chr(extra_code)
            extra_code += 1

    return b2u


def build_pair_freqs(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """全量统计 pair 频次，仅在训练开始前调用一次"""
    pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(len(token_ids) - 1):
        pair_freqs[(token_ids[i], token_ids[i + 1])] += 1
    return pair_freqs


def resolve_tokenizer_path(path: str) -> Path:
    """配置中的相对路径统一按 tokenizer 目录解析。"""
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def serialize_bpe_token(token_bytes: bytes, b2u: dict[int, str]) -> str:
    return "".join(b2u[byte] for byte in token_bytes)


def validate_artifacts(
    vocab_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    special_token_ids: set[int],
) -> None:
    with vocab_path.open("r", encoding="utf-8") as f:
        serialized_vocab = json.load(f)

    if len(serialized_vocab) != vocab_size:
        raise ValueError(f"vocab size mismatch: expected {vocab_size}, got {len(serialized_vocab)}")

    id_to_token = {token_id: token for token, token_id in serialized_vocab.items()}
    b2u = bytes_to_unicode()
    for byte in range(256):
        actual = id_to_token.get(byte)
        expected = b2u[byte]
        if actual != expected:
            raise ValueError(f"base byte {byte} serialized as {actual!r}, expected {expected!r}")

    for token in special_tokens:
        token_id = serialized_vocab.get(token)
        if token_id not in special_token_ids:
            raise ValueError(f"special token {token!r} has invalid id {token_id!r}")


def load_training_text(
    data_path: Path,
    max_chars: int | None,
    seed: int,
    random_sample: bool,
) -> str:
    if max_chars is None:
        return data_path.read_text(encoding="utf-8")

    if max_chars <= 0:
        raise ValueError("max_chars must be positive or null")

    if not random_sample:
        with data_path.open("r", encoding="utf-8") as f:
            return f.read(max_chars)

    rng = random.Random(seed)
    selected_lines: list[str] = []
    selected_chars = 0

    with data_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line_len = len(line)
            if line_len > max_chars:
                line = line[:max_chars]
                line_len = len(line)

            if selected_chars + line_len <= max_chars:
                selected_lines.append(line)
                selected_chars += line_len
                continue

            replacement_index = rng.randrange(line_number)
            if replacement_index < len(selected_lines):
                old_line = selected_lines[replacement_index]
                new_total = selected_chars - len(old_line) + line_len
                if new_total <= max_chars:
                    selected_lines[replacement_index] = line
                    selected_chars = new_total

    rng.shuffle(selected_lines)
    return "".join(selected_lines)


def merge_and_update(
    token_ids:  list[int],
    best_pair:  tuple[int, int],
    new_id:     int,
    pair_freqs: dict[tuple[int, int], int],
) -> list[int]:
    """
    把 token_ids 中所有 best_pair 替换为 new_id，
    同时增量更新 pair_freqs（只改动受合并影响的左右邻居）。

    合并位置示意：
        ... left  [a  b]  right ...
              ↓  合并后  ↓
        ... left   c     right ...

    受影响的 pair：
        (left, a)  → 消失，改为 (left, c)
        (b, right) → 消失，改为 (c, right)
        (a, b)     → 消耗一次
    """
    a, b = best_pair
    new_ids: list[int] = []
    i = 0
    n = len(token_ids)

    while i < n:
        is_match = (
            i < n - 1
            and token_ids[i]     == a
            and token_ids[i + 1] == b
        )

        if is_match:
            # --- 更新左邻 pair ---
            if new_ids:
                left = new_ids[-1]
                old_pair = (left, a)
                new_pair = (left, new_id)
                pair_freqs[old_pair] -= 1
                pair_freqs[new_pair] += 1

            # --- 更新右邻 pair ---
            if i + 2 < n:
                right = token_ids[i + 2]
                old_pair = (b, right)
                new_pair = (new_id, right)
                pair_freqs[old_pair] -= 1
                pair_freqs[new_pair] += 1

            # --- best_pair 本身消耗一次 ---
            pair_freqs[(a, b)] -= 1

            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(token_ids[i])
            i += 1

    return new_ids


# ─────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────

def train_bpe_tokenizer():
    config_path = SCRIPT_DIR / "tokenizer_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_size         = config.get("vocab_size",         6000)
    data_path          = resolve_tokenizer_path(config.get("data_path",          "tokenizer_data.txt"))
    vocab_output_path  = resolve_tokenizer_path(config.get("vocab_output_path",  "vocab.json"))
    merges_output_path = resolve_tokenizer_path(config.get("merges_output_path", "merges.txt"))
    special_tokens     = config.get("special_tokens",     [])
    max_chars          = config.get("max_chars",          10_000_000)
    progress_interval  = config.get("progress_interval",  100)
    random_sample      = config.get("random_sample",      True)
    sample_seed        = config.get("sample_seed",        42)
    reserved_special_bytes = {token.encode("utf-8") for token in special_tokens}

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")
    if len(special_tokens) != len(set(special_tokens)):
        raise ValueError("special_tokens contains duplicates")

    print("=" * 55)
    print("BPE Tokenizer 训练")
    print("=" * 55)
    print(f"  词表大小       : {vocab_size}")
    print(f"  训练数据路径   : {data_path}")
    print(f"  最大字符数     : {'全部' if max_chars is None else f'{max_chars:,}'}")
    print(f"  随机采样       : {random_sample}")
    print(f"  采样 seed      : {sample_seed}")
    print(f"  进度打印间隔   : {progress_interval}")
    print(f"  special tokens : {special_tokens}")
    print(f"  词表输出路径   : {vocab_output_path}")
    print(f"  合并规则路径   : {merges_output_path}")
    print("=" * 55)

    # ── 1. 读取训练数据 ──────────────────────────
    t0 = time.time()
    print(f"\n[1/4] 读取训练数据...")
    data = load_training_text(data_path, max_chars, sample_seed, random_sample)

    sampled = max_chars is not None and len(data) >= max_chars
    sample_label = "随机采样" if random_sample and max_chars is not None else "顺序读取"
    print(f"  字符数（读取后） : {len(data):,}  （{sample_label}{'，达到上限' if sampled else ''}）")
    print(f"  耗时             : {time.time() - t0:.2f}s")

    # ── 2. 初始化 vocab（只放 256 个 base bytes，special tokens 最后加）──
    print(f"\n[2/4] 初始化词表...")
    vocab: dict[bytes, int] = {}

    for i in range(256):
        vocab[bytes([i])] = i

    id_to_bytes: dict[int, bytes] = {v: k for k, v in vocab.items()}

    print(f"  base vocab（字节）: 256")
    print(f"  special tokens 将在训练完成后追加到末尾: {special_tokens}")

    # ── 3. 语料 → token id 列表 ──────────────────
    data_bytes: bytes     = data.encode("utf-8")
    token_ids:  list[int] = list(data_bytes)

    print(f"  UTF-8 字节数        : {len(data_bytes):,}")
    print(f"  初始 token 序列长度 : {len(token_ids):,}")

    # ── 4. 一次性初始化 pair 频次表 ──────────────
    print(f"\n[3/4] 初始化 pair 频次表（仅此一次）...")
    t_freq = time.time()
    pair_freqs = build_pair_freqs(token_ids)
    print(f"  不同 pair 数 : {len(pair_freqs):,}")
    print(f"  耗时         : {time.time() - t_freq:.2f}s")

    # ── 5. BPE 合并循环 ──────────────────────────
    # special tokens 放末尾，BPE 合并从 256 开始，跑满 vocab_size - len(special_tokens) - 256 步
    num_merges  = vocab_size - len(special_tokens) - len(vocab)
    merges: list[tuple[int, int]] = []
    b2u = bytes_to_unicode()

    print(f"\n  开始合并，共 {num_merges} 步")
    print(f"  BPE id 范围: 256 ~ {256 + num_merges - 1}")
    print(f"  special token id 范围: {256 + num_merges} ~ {vocab_size - 1}")
    print("-" * 75)

    train_start = time.time()

    for step in range(num_merges):
        step_start = time.time()

        best_pair = max(
            (
                p for p in pair_freqs
                if pair_freqs[p] > 0
                and id_to_bytes[p[0]] + id_to_bytes[p[1]] not in reserved_special_bytes
            ),
            key=pair_freqs.get,
            default=None,
        )
        if best_pair is None:
            print("  所有 pair 频次已归零，提前结束。")
            break

        best_freq = pair_freqs[best_pair]

        # 分配新 id，更新 vocab
        new_id    = len(vocab)
        new_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        vocab[new_bytes]    = new_id
        id_to_bytes[new_id] = new_bytes
        merges.append(best_pair)

        # 合并 + 增量更新 pair_freqs
        token_ids = merge_and_update(token_ids, best_pair, new_id, pair_freqs)

        # 每 200 步清理频次为 0 的残留 pair
        if (step + 1) % 200 == 0:
            before = len(pair_freqs)
            pair_freqs = defaultdict(int, {k: v for k, v in pair_freqs.items() if v > 0})
            print(f"  [清理] pair_freqs: {before:,} → {len(pair_freqs):,}")

        # 进度打印
        token_unicode = "".join(b2u[byte] for byte in new_bytes)
        try:
            token_utf8 = new_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_utf8 = "".join(f"\\x{b:02x}" for b in new_bytes)

        if step == 0 or (step + 1) % progress_interval == 0:
            elapsed   = time.time() - train_start
            remaining = elapsed / (step + 1) * (num_merges - step - 1)
            print(
                f"  步骤 {step+1:>5}/{num_merges}"
                f"  新token(id={new_id}): {token_utf8!r:<16} ({token_unicode})"
                f"  频次={best_freq:>8,}"
                f"  序列长={len(token_ids):>11,}"
                f"  步耗时={time.time()-step_start:.3f}s"
                f"  剩余≈{remaining/60:.1f}min"
            )
        elif progress_interval <= 1:
            print(f"  步骤 {step+1:>5}/{num_merges}  新token(id={new_id}): {token_utf8!r:<16} ({token_unicode})  频次={best_freq:>8,}")

    total_time = time.time() - train_start
    print("-" * 75)
    print(f"  BPE 训练完成！实际合并步数 : {len(merges)}")
    print(f"  BPE vocab 大小             : {len(vocab)}  (id 0 ~ {len(vocab)-1})")
    print(f"  最终序列长度               : {len(token_ids):,}")
    print(f"  压缩比                     : {len(data_bytes) / len(token_ids):.2f}x")
    print(f"  总训练耗时                 : {total_time/60:.1f} min")

    # ── 6. 追加 special tokens（放在 BPE vocab 末尾）──
    print(f"\n  追加 special tokens...")
    special_token_ids: set[int] = set()
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        if token_bytes in vocab:
            raise ValueError(f"special token collides with BPE token bytes: {token_str!r}")
        new_id = len(vocab)
        vocab[token_bytes]    = new_id
        id_to_bytes[new_id]   = token_bytes
        special_token_ids.add(new_id)
        print(f"    id={new_id}  {token_str}")

    print(f"  最终词表大小（BPE + special）: {len(vocab)}")

    # ── 7. 保存产物 ──────────────────────────────
    print(f"\n[4/4] 保存产物...")
    os.makedirs(vocab_output_path.parent, exist_ok=True)
    os.makedirs(merges_output_path.parent, exist_ok=True)

    # vocab.json：BPE token 用 bytes_to_unicode 表示，special token 直接用原始字符串
    vocab_serializable = {}
    for token_bytes, token_id in vocab.items():
        try:
            if token_id in special_token_ids:
                token_str = token_bytes.decode("utf-8")
                vocab_serializable[token_str] = token_id
            else:
                vocab_serializable[serialize_bpe_token(token_bytes, b2u)] = token_id
        except UnicodeDecodeError:
            vocab_serializable[serialize_bpe_token(token_bytes, b2u)] = token_id

    with vocab_output_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    print(f"  vocab   → {vocab_output_path}  ({len(vocab)} 条)")

    with merges_output_path.open("w", encoding="utf-8") as f:
        for id_a, id_b in merges:
            f.write(f"{id_a} {id_b}\n")
    print(f"  merges  → {merges_output_path}  ({len(merges)} 条)")

    validate_artifacts(vocab_output_path, vocab_size, special_tokens, special_token_ids)
    print("  产物校验通过")

    print("\n全部完成 ✓")


if __name__ == "__main__":
    print("这是一个BPE训练的脚本")
    train_bpe_tokenizer()
