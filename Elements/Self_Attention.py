"""
이 코드는 셀프 어텐션 메커니즘의 기본적인 구현 방법을 보여줍니다. 
쿼리, 키, 밸류를 초기화하고, 스케일링된 도트 프로덕트 어텐션을 통해 새로운 값 벡터를 계산합니다. 
마스킹을 추가하여 디코더에서 미래의 단어 정보를 참조하지 못하도록 합니다. 
마지막으로, 계산된 새로운 값 벡터와 어텐션 맵을 출력하여, 셀프 어텐션의 결과를 확인할 수 있습니다.
"""


# 필요한 라이브러리를 임포트합니다.
import numpy as np
import math

# 레이어(Layer), 차원(Dimension) 및 값(Value)의 크기를 정의합니다.
L, d_k, d_v = 4, 8, 8

# 쿼리(Query), 키(Key), 밸류(Value)를 무작위로 초기화합니다.
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

# 생성된 쿼리, 키, 밸류를 출력합니다.
print("Q\n", q)
print("K\n", k)
print("V\n", v)

# Scaled Dot-Product Attention 구현
# 쿼리와 키의 전치행렬(T)을 곱하고, 차원의 제곱근으로 나누어 스케일링합니다.
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = softmax(scaled)
    output = np.matmul(attention, v)
    return output, attention

# 마스킹을 통해 미래의 단어로부터 어텐션을 방지합니다.
# 디코더에서 사용됩니다. 인코더에서는 필요하지 않습니다.
mask = np.tril(np.ones((L, L)))
mask[mask == 0] = -np.infty
mask[mask == 1] = 0

# 어텐션 메커니즘과 마스킹을 적용하여 새로운 밸류 벡터를 계산합니다.
values, attention = scaled_dot_product_attention(q, k, v, mask=mask)

# 새로 계산된 밸류와 어텐션 맵을 출력합니다.
print("New V\n", values)
print("Attention\n", attention)
