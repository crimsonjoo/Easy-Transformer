"""
트랜스포머 모델에서 중요한 구성 요소인 포지셔널 인코딩을 생성합니다.
"""


# 필요한 라이브러리를 불러옵니다.
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    트랜스포머 모델에서 사용하는 포지셔널 인코딩을 생성하는 클래스입니다.
    이 클래스는 모델의 각 단어 위치에 대한 고유한 인코딩을 생성하여,
    모델이 단어의 순서 정보를 활용할 수 있도록 돕습니다.
    
    파라미터:
    - d_model: 임베딩 차원의 크기
    - max_sequence_length: 입력 시퀀스의 최대 길이
    """
    
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        # 짝수 인덱스(even)에 해당하는 위치값을 계산합니다.
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)

        # 포지션 값(position)을 계산합니다. 각 위치에 대한 인코딩을 생성하기 위해 사용됩니다.
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)

        # 짝수 인덱스에 해당하는 포지셔널 인코딩 값을 계산합니다.
        even_PE = torch.sin(position / denominator)
        
        # 홀수 인덱스(odd)에 해당하는 포지셔널 인코딩 값을 계산합니다.
        odd_PE = torch.cos(position / denominator)

        # 짝수 및 홀수 포지셔널 인코딩 값을 쌓아 전체 포지셔널 인코딩을 생성합니다.
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        
        # 생성된 포지셔널 인코딩 값을 2차원으로 펼쳐 최종 출력값을 생성합니다.
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        
        return PE

if __name__ == "__main__":
    # 포지셔널 인코딩을 생성하기 위한 설정값을 정의합니다.
    d_model = 6
    max_sequence_length = 10
    
    # 포지셔널 인코딩 객체를 생성하고, 포워드 메소드를 호출하여 포지셔널 인코딩 값을 계산합니다.
    pe = PositionalEncoding(d_model=d_model, max_sequence_length=max_sequence_length)
    positional_encoding = pe.forward()
    
    # 계산된 포지셔널 인코딩 값을 출력합니다.
    print(positional_encoding)
