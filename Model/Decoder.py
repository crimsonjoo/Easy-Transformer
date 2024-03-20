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


# 위치별 전결합 피드포워드 네트워크
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 레이어 정규화
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True) + self.eps
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


# 멀티헤드 어텐션
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model의 차원 크기는 반드시 num_heads의 배수"
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x).view(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        q, k, v = qkv
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


# 디코더 레이어
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_mask):
        attention = self.self_attention(x, mask=self_mask)
        x = self.norm1(x + self.dropout1(attention))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


# 디코더 클래스
class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        self.norm = LayerNormalization([d_model])

    def forward(self, x, self_mask):
        for layer in self.layers:
            x = layer(x, self_mask)
        x = self.norm(x)
        return x
