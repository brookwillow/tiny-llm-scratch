import torch
import math
from torch import nn
from einops import rearrange

class ModelConfig:
    def __init__(self, vocab_size: int = 6400, hidden_size: int = 768, num_heads: int = 12, num_layers: int = 12, max_seq_len: int = 512, 
                 device = None, dtype = torch.float32):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        # SwiGLU 惯例: 8/3 * hidden_size ≈ 2048
        self.hidden_size_ff = int(8 / 3 * hidden_size)  
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

class Linear(nn.Module):
    '''
    线性层
    '''
    def __init__(self, in_features: int, out_features: int, device= None, dtype= None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        kwargs = {"device": device, "dtype": dtype}      
        # 此处权重矩阵按照（out,in）排列为了 CPU/GPU 缓存高效读取，单个输出神经元的所有权重必须在内存连续排布
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **kwargs)) 
        # Xavier 权重初始化，实现截断正态分布初始化
        std =  (2/(in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean= 0, std=std, a= -3*std, b= 3*std)

    def forward(self, X):
        return torch.einsum("...i, oi -> ...o", X, self.weight)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class Embedding(nn.Module):
    '''
    嵌入层
    '''
    def __init__(self, vocab_size: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 使用parameter封装权重矩阵， 因为需要通过优化器更新，且随模型保存
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype))

        # 初始化
        # 均值为0，标准差为1，阶段范围[-3,3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, X):
        return self.weight[X]

    def extra_repr(self):
        return f'vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}'


class RoPEEmbedding(nn.Module):
    '''
    定义RoPE位置编码
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        # 初始化RoPE位置编码参数

        self.d_k = d_k
        # 构建 [0, 2, 4, ..., d_k - 2]/ d_k 序列
        powers = torch.arange(0, d_k, 2, device = device).float() / d_k
        # 构建分母部分
        freqs = 1.0 / (theta ** powers)

        # 创建位置的序列
        t = torch.arange(max_seq_len, device = device).float()

        # (max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def extra_repr(self):
        return f'd_k={self.d_k}, max_seq_len={self.cos_cached.shape[0]}'


    def forward(self, X, token_position):
        # 实现RoPE位置编码的前向传播
        cos = self.cos_cached[token_position]
        sin = self.sin_cached[token_position]

        if X.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        cos = cos.to(X.dtype)
        sin = sin.to(X.dtype)

        X_even = X[..., 0::2]
        X_odd = X[..., 1::2]

        output = torch.empty_like(X)
        output[..., 0::2] = X_even * cos - X_odd * sin
        output[..., 1::2] = X_even * sin + X_odd * cos

        return output


class RMSNorm(nn.Module):
    '''
    定义RSNorm归一化层
    '''
    def __init__(self, hidden_size: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        kwargs = {"device": device, "dtype" : dtype}
        self.W = nn.Parameter(torch.ones(hidden_size, ** kwargs))
        self.eps = eps

    def extra_repr(self):
        return f'hidden_size={self.W.shape[0]}, eps={self.eps}'

    def forward(self, X):
        # （batch_size, seq_len, hidden_size）
        in_dtype = X.dtype
        X_float = X.to(torch.float32)

        # 计算RMS Root Mean Square 平方根
        # rms = sqart(mean(X^2) + eps)

        ms = X_float.pow(2).mean(dim = -1, keepdim = True)
        rms = torch.sqrt(ms + self.eps)

        result = (X_float / rms) * self.W
        return result.to(in_dtype)


class TransformerBlock(nn.Module):
    '''
    定义Transformer Block
    '''
    def __init__(self, hidden_size:int, num_head: int, hidden_size_ff:int, max_seq_len: int = 512, device = None, dtype = None):
        super().__init__()
        # 初始化Transformer Block的各个组件
        self.num_head = num_head
        self.d_k = hidden_size // num_head
        self.layer_norm1 = RMSNorm(hidden_size, device = device, dtype = dtype)
        self.layer_norm2 = RMSNorm(hidden_size, device = device, dtype = dtype)
        self.rope = RoPEEmbedding(theta=10000.0, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        self.q_proj = Linear(hidden_size, hidden_size, device= device, dtype= dtype)
        self.k_proj = Linear(hidden_size, hidden_size, device= device, dtype= dtype)
        self.v_proj = Linear(hidden_size, hidden_size, device= device, dtype= dtype)

        self.out_proj = Linear(hidden_size, hidden_size, device = device, dtype = dtype)


        self.ffn = SwiGLU(hidden_size, hidden_size_ff, device, dtype)

    def forward(self, X, block_idx=0):
        # 将输入形状记录下来
        batch_size, seq_len, hidden_size = X.shape
        device = X.device

        # 保存残差
        residual = X

        # 进行层归一化
        X = self.layer_norm1(X)

        # 经过线性矩阵的转换
        X_Q = self.q_proj(X)
        X_K = self.k_proj(X)
        X_V = self.v_proj(X)

        # 使用einops对QKV矩阵先切分head, 然后转换成 (batch_size, head_num, seq_len, dim)
        X_Q = rearrange(X_Q, "... s (h d) -> ... h s d", h = self.num_head)
        X_K = rearrange(X_K, "... s (h d) -> ... h s d", h = self.num_head)
        X_V = rearrange(X_V, "... s (h d) -> ... h s d", h = self.num_head)

        # 对Q,K矩阵进行RoPE
        token_position = torch.arange(seq_len, device = device).expand(batch_size, seq_len)
        X_Q = self.rope(X_Q, token_position)
        X_K = self.rope(X_K, token_position)

        # 计算注意力分数，缩放因子是 sqrt(d_k)
        scores = torch.einsum("...nk, ...mk -> ...nm", X_Q, X_K) / math.sqrt(self.d_k)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device = device, dtype = torch.bool))
        scores = scores.masked_fill(mask == False, float("-inf"))

        # softmax
        probs = torch.softmax(scores, dim = -1)

        attn_out = torch.einsum("...nm, ...mk -> ...nk", probs, X_V)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")

        attn_out = self.out_proj(attn_out)

        # 残差连接
        X = residual + attn_out

        # FFN with residual
        X = X + self.ffn(self.layer_norm2(X))

        return X



class SwiGLU(nn.Module):
    '''
    定义前馈神经网络
    '''
    def __init__(self, hidden_size:int, hidden_size_ff: int, device = None, dtype = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_ff = hidden_size_ff

        # W1, W3 并行的升维矩阵 hidden_size -> hidden_size_ff
        self.W1 = Linear(hidden_size, hidden_size_ff, device, dtype)
        self.W3 = Linear(hidden_size, hidden_size_ff, device, dtype)

        # W2 降维矩阵，hidden_size_ff -> hidden_size
        self.W2 = Linear(hidden_size_ff, hidden_size, device, dtype)

    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, hidden_size_ff={self.hidden_size_ff}'



    def forward(self, X):
        gate = torch.nn.functional.silu(self.W1(X))
        signal = self.W3(X)

        return self.W2(gate * signal)

class Model(nn.Module):
    ''' 
    定义模型结构
    '''
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        # self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        # transformer block
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.hidden_size, config.num_heads, config.hidden_size_ff, config.max_seq_len, config.device, config.dtype) for _ in range(config.num_layers)])
        # 最终归一化层
        self.final_norm = RMSNorm(config.hidden_size, device=config.device, dtype=config.dtype)
        # 输出层
        self.output_layer = Linear(config.hidden_size, config.vocab_size, device=config.device, dtype=config.dtype)

    def forward(self, X):
        # {batch_size, seq_len}
        batch_size, seq_len = X.shape

        # 嵌入层, {batch_size, seq_len, hidden_size}
        X = self.embedding(X)

        # transformer block, {batch_size, seq_len, hidden_size}
        for i, block in enumerate(self.transformer_blocks):
            X = block(X, block_idx=i)
        # 最终归一化
        X = self.final_norm(X)
        # 输出投影, {batch_size, seq_len, vocab_size}
        X = self.output_layer(X)

        return X

if __name__ == "__main__":
    config = ModelConfig()
    model = Model(config)
    X = torch.randint(0, config.vocab_size, (2, 10))  # 假设输入是一个batch_size为2，序列长度为10的随机整数张量
    output = model(X)
    print(model)