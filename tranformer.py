import torch
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性层用于生成 Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        Q, K, V 的形状: (batch_size, num_heads, seq_len, d_k)
        mask 的形状: (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        mask 中为 1 的位置允许被看到，为 0 的位置会被掩蔽（填充或未来位置）。
        """
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 应用 softmax 得到注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 计算加权和
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入张量拆分为多个头
        输入形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_lenth, d_model = x.size()
        return x.view(batch_size, seq_lenth, self.num_heads, self.d_k).transpose(1, 2)  # 转换为 (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        """
        将多个头的输出合并为一个张量
        输入形状: (batch_size, num_heads, seq_len, d_k)
        输出形状: (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        多头注意力的完整前向流程
        query, key, value 的形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, seq_len, d_model)
        """
        # 1. 生成 Q, K, V 并拆分为多头
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # 2. 缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头, 做最后的线性变换
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



# --- 编码器核心层 ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# --- 编码器: 堆叠 N 个 EncoderLayer ---
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # x 形状: (batch_size, src_seq_len, d_model)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # 形状: (batch_size, src_seq_len, d_model)


# --- 解码器核心层 ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output,
                                            encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# --- 解码器: 堆叠 N 个 DecoderLayer ---
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x 形状: (batch_size, tgt_seq_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # 形状: (batch_size, tgt_seq_len, d_model)


# --- 掩码工具 ---
def make_src_mask(src, pad_idx):
    """
    编码器 padding 掩码: 真实 token 位置为 1, padding 位置为 0
    返回形状: (batch_size, 1, 1, src_seq_len)
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx):
    """
    解码器掩码 = 因果(下三角)掩码 AND padding 掩码
    让第 i 个位置只能看到前 i 个 token (防止偷看未来)
    返回形状: (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (b, 1, 1, tgt_seq_len)
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len)).bool()   # (tgt_seq_len, tgt_seq_len)
    causal = causal.unsqueeze(0).unsqueeze(0)                  # (1, 1, tgt_seq_len, tgt_seq_len)
    return causal & tgt_pad_mask


# --- 完整 Transformer ---
class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    输入/输出流程: 词嵌入 -> 位置编码 -> Encoder -> Decoder -> 线性投影
    """
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout, num_encoder_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src 形状: (batch_size, src_seq_len), tgt 形状: (batch_size, tgt_seq_len)
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.fc_out(output)  # 形状: (batch_size, tgt_seq_len, tgt_vocab_size)
