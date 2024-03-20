import torch
import math
from torch import nn
import torch.nn.functional as F



# 스케일드 닷 프로덕트 어텐션 함수
def scaled_dot_product(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention 구현.
    :param q: 쿼리 텐서
    :param k: 키 텐서
    :param v: 값 텐서
    :param mask: 마스크 텐서 (옵션)
    :return: attention 값을 적용한 값 텐서와 attention 텐서
    """
    d_k = q.size()[-1]  # 키 벡터의 차원
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # 스케일링
    if mask is not None:
        scaled += mask  # 마스크가 있으면 추가
    attention = F.softmax(scaled, dim=-1)  # 소프트맥스를 통한 어텐션 가중치 계산
    values = torch.matmul(attention, v)  # 가중치를 값 벡터에 적용
    return values, attention


# 멀티 헤드 어텐션 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)  # Q, K, V를 위한 선형 변환
        self.linear_layer = nn.Linear(d_model, d_model)  # 최종 선형 변환

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_layer(x).view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = [tensor.permute(0, 2, 1, 3) for tensor in (q, k, v)]  # 헤드 차원을 앞으로 이동
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.linear_layer(values)


# 레이어 노멀라이제이션 클래스
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# 포지션 와이즈 피드포워드 네트워크 클래스
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)


# 인코더 레이어 클래스
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x + self.self_attn(x, mask))
        return self.norm2(x2 + self.ffn(x2))


# 인코더 클래스
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)