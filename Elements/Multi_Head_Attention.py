"""
이 코드는 멀티 헤드 어텐션을 구현하고 간단한 입력에 대해 테스트하는 예제입니다. 
멀티 헤드 어텐션은 어텐션 메커니즘을 여러 번 병렬로 수행하여, 서로 다른 표현 공간에서 입력 데이터의 정보를 집약합니다. 
이를 통해 모델이 다양한 측면에서 입력 데이터를 이해할 수 있게 돕습니다. 
"""


import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

# Multi-Head Attention 클래스 정의
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        """
        input_dim: 입력 벡터의 차원
        d_model: 모델의 차원 (임베딩 벡터의 크기)
        num_heads: 헤드의 수
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Q, K, V 값을 생성하기 위한 선형 변환
        self.qkv_layer = nn.Linear(input_dim , 3 * d_model)
        # 최종 출력을 위한 선형 변환
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        # 멀티 헤드 어텐션을 위한 차원 재배치
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        # Q, K, V 분리
        q, k, v = qkv.chunk(3, dim=-1)
        # Scaled Dot-Product Attention 수행
        values, attention = scaled_dot_product(q, k, v, mask)
        # 멀티 헤드 결과 병합
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # 최종 출력 생성
        out = self.linear_layer(values)
        return out

def scaled_dot_product(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention 구현
    """
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# 멀티 헤드 어텐션을 테스트하기 위한 임의의 입력 데이터 생성
batch_size = 1
sequence_length = 4
input_dim = 512
d_model = 512
num_heads = 8

x = torch.randn((batch_size, sequence_length, input_dim))
model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)

print("Output shape:", out.shape)
