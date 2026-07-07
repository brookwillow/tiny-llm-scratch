import json
import time
from pathlib import Path

class BPETokenizer:
    def __init__(self, vocab_path: str, merges_path: str):
        self.vocab = self._load_vocab(vocab_path)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = self._load_merges(merges_path)
        self.merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        self.merge_to_id = {merge: 256 + rank for rank, merge in enumerate(self.merges)}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def encode(self, text: str):
        token_ids: list[int] = list(text.encode("utf-8"))

        while len(token_ids) > 1:
            best_pair = self._get_best_pair(token_ids)
            if best_pair is None:
                break
            token_ids = self._merge_pair(token_ids, best_pair, self.merge_to_id[best_pair])

        return token_ids

    def decode(self, token_ids: list[int]):
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        text = "".join(tokens)
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode("utf-8", errors="replace")

    def _get_best_pair(self, token_ids: list[int]):
        best_pair = None
        best_rank = float("inf")

        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            rank = self.merge_ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_pair = pair
                best_rank = rank

        return best_pair

    def _merge_pair(self, token_ids: list[int], pair: tuple[int, int], new_id: int):
        merged_ids: list[int] = []
        i = 0

        while i < len(token_ids):
            if (
                i < len(token_ids) - 1
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                merged_ids.append(new_id)
                i += 2
            else:
                merged_ids.append(token_ids[i])
                i += 1

        return merged_ids

    def _load_vocab(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab

    def _load_merges(self, merges_path: str):
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        return [tuple(int(x) for x in line.split()) for line in lines]

    def _bytes_to_unicode(self):
        """GPT-2 style byte-to-unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))



if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    vocab_path = base_dir / "vocab.json"
    merges_path = base_dir / "merges.txt"
    tokenizer = BPETokenizer(vocab_path, merges_path)

    test_cases = [
        ("en_short", "Hello, world! This is a test of the BPE tokenizer."),
        ("zh_short", "你好，世界！这是一个测试。我们正在观察 tokenizer 的编码速度。"),
        ("mixed_short", "今天 temperature is 28.5°C, humidity is 72%, 风有点大。"),
        ("code_short", "def merge_lists(a, b):\n    return sorted(set(a + b))\n"),
        ("multilingual", "こんにちは、世界！안녕하세요, 세계! Привет, мир! مرحبا بالعالم!"),
        (
            "zh_medium",
            "大模型训练通常包括数据清洗、分词、预训练、监督微调和评估几个阶段。"
            "在这个项目里，我们希望从 tokenizer、模型结构、优化器到训练循环都尽量自己实现，"
            "这样可以更清楚地理解每一个模块的输入输出和性能瓶颈。",
        ),
        (
            "en_medium",
            "Tokenization is often treated as a preprocessing detail, but it has a direct impact "
            "on context length, training throughput, memory usage, and the distribution of symbols "
            "that the language model needs to learn.",
        ),
        (
            "code_medium",
            "class TinyModel:\n"
            "    def __init__(self, vocab_size, hidden_size):\n"
            "        self.vocab_size = vocab_size\n"
            "        self.hidden_size = hidden_size\n\n"
            "    def forward(self, input_ids):\n"
            "        # embedding -> transformer blocks -> logits\n"
            "        return input_ids\n",
        ),
        (
            "zh_long",
            (
                "我们现在用一个较长的中文段落来测试编码耗时。"
                "这段文本会重复出现一些常见词，也会混入少量不常见词汇，例如拓扑绝缘体、量子纠缠、"
                "CRISPR-Cas9 基因编辑和分布式优化。"
                "如果 tokenizer 的 merge 规则覆盖了常见中文片段，那么 token 数量应该会明显少于 UTF-8 字节数。"
            )
            * 8,
        ),
        (
            "mixed_long",
            (
                "User: 请解释一下 attention mechanism and rotary positional embedding.\n"
                "Assistant: Attention computes weighted combinations of values based on query-key similarity. "
                "RoPE injects position information by rotating query and key vectors in each attention head.\n"
                "代码示例: scores = torch.einsum('...qd,...kd->...qk', q, k) / sqrt(d)\n"
            )
            * 8,
        ),
    ]

    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_encode_time = 0.0

    print(f"{'case':<14} {'chars':>8} {'bytes':>8} {'tokens':>8} {'ms':>10} {'tok/s':>12} {'byte/tok':>10} status")
    print("-" * 92)
    for name, text in test_cases:
        start = time.perf_counter()
        token_ids = tokenizer.encode(text)
        encode_time = time.perf_counter() - start
        decoded_text = tokenizer.decode(token_ids)
        status = "OK" if decoded_text == text else "FAIL"

        char_count = len(text)
        byte_count = len(text.encode("utf-8"))
        token_count = len(token_ids)
        token_per_second = token_count / encode_time if encode_time > 0 else 0.0
        bytes_per_token = byte_count / token_count if token_count > 0 else 0.0

        total_chars += char_count
        total_bytes += byte_count
        total_tokens += token_count
        total_encode_time += encode_time

        print(
            f"{name:<14} {char_count:>8} {byte_count:>8} {token_count:>8} "
            f"{encode_time * 1000:>10.2f} {token_per_second:>12.1f} "
            f"{bytes_per_token:>10.3f} {status}"
        )

    print("-" * 92)
    total_token_per_second = total_tokens / total_encode_time if total_encode_time > 0 else 0.0
    total_bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0.0
    print(
        f"{'total':<14} {total_chars:>8} {total_bytes:>8} {total_tokens:>8} "
        f"{total_encode_time * 1000:>10.2f} {total_token_per_second:>12.1f} "
        f"{total_bytes_per_token:>10.3f}"
    )
