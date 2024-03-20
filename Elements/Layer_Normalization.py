"""
이 파일은 레이어 정규화의 기본적인 이해를 돕기 위해 설계되었습니다. 
레이어 정규화는 딥러닝 모델에서 학습의 안정성과 수렴 속도를 개선하기 위해 널리 사용됩니다. 
이 코드 예제를 통해 사용자는 레이어 정규화가 입력 데이터의 평균을 0으로, 
표준편차를 1로 조정하여 데이터를 정규화하는 과정을 이해할 수 있습니다. 
"""


# 필요한 라이브러리를 임포트합니다.
import torch
from torch import nn

# LayerNormalization 클래스를 정의합니다.
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        """
        레이어 정규화를 위한 클래스 초기화
        
        :param parameters_shape: 정규화할 파라미터의 형태
        :param eps: 분모가 0이 되는 것을 방지하기 위한 작은 값
        """
        super().__init__()
        self.eps = eps
        # 감마와 베타를 학습가능한 파라미터로 정의합니다. 초기값은 감마가 1, 베타가 0입니다.
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        """
        레이어 정규화를 수행하는 forward 메서드
        
        :param inputs: 정규화할 입력
        """
        # 입력값으로부터 평균과 표준편차를 계산합니다.
        mean = inputs.mean((-1), keepdim=True)
        std = inputs.std((-1), keepdim=True) + self.eps
        
        # 레이어 정규화를 수행합니다.
        y = (inputs - mean) / std
        return self.gamma * y + self.beta

# 테스트를 위한 입력 데이터를 생성합니다.
inputs = torch.tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]], dtype=torch.float32)

# LayerNormalization 인스턴스를 생성합니다.
layer_norm = LayerNormalization(inputs.size()[-1])

# forward 메서드를 호출하여 레이어 정규화를 수행합니다.
normalized_inputs = layer_norm(inputs)

print("정규화된 입력:\n", normalized_inputs)