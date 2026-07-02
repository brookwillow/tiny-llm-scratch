import json
from pathlib import Path

class BPETokenizer:
    def __init__(self, vocab_path: str, merges_path: str):
        self.vocab = self._load_vocab(vocab_path)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = self._load_merges(merges_path)
        # byte <-> unicode 映射（GPT-2 风格）
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(self.merges)}
    
    def encode(self, text: str):
        tokens = text.encode("utf-8")
        token_ids:list[int] = list(tokens)
        merge_count = 0
        # 按照merges顺序进行合并
        for merge in self.merges:
            pair = tuple(int(x) for x in merge.split())
            # 反复合并直到没有相邻匹配
            i = 0
            new_ids = []
            while i < len(token_ids):
                if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i+1] == pair[1]:
                    # 合并为新 token（查 vocab 获取合并后的 id）
                    merged_token = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
                    new_id = self.vocab.get(merged_token)
                    if new_id is not None:
                        new_ids.append(new_id)
                        i += 2
                        merge_count += 1
                    else:
                        new_ids.append(token_ids[i])
                        i += 1
                else:
                    new_ids.append(token_ids[i])
                    i += 1
            token_ids = new_ids
    
        return token_ids
    

    def decode(self, token_ids: list[int]):
        # 将BPE tokens解码为原始文本
        tokens = [self.id_to_token.get(id, '') for id in token_ids]
        text = ''.join(tokens)
        # 将 GPT-2 unicode 字符映射回原始字节
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode("utf-8", errors="replace")


    def _load_vocab(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    

    def _load_merges(self, merges_path: str):
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges = [line.strip() for line in f if line.strip()]
        return merges


    def _bytes_to_unicode(self):
        """GPT-2 style byte-to-unicode mapping."""
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
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

    test_lists = [
        "Hello, world!",
        "This is a test.",
        "你哈好，世界！",
        "こんにちは、世界！",
        "안녕하세요, 세계!",
        "Привет, мир!",
        "مرحبا بالعالم!",
        "Bonjour le monde!",
        "¡Hola, mundo!",
        "this is a test of the BPE tokenizer.",
        "这是一个测试。我们是一家人，给我一个拥抱吧！",
    ]
    for text in test_lists:
        print(f"Original text: {text}")
        token_ids = tokenizer.encode(text)
        print(f"Encoded token IDs: {token_ids}")
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded text: {decoded_text}")
        print("-" * 50)
