"""
이 스크립트는 문장을 토큰화하고, 각 토큰을 인덱스로 변환한 후, 임베딩 벡터로 변환하는 과정을 담고 있습니다. 
추가적으로 포지셔널 인코딩을 적용하여, 모델이 단어의 순서 정보를 활용할 수 있게 합니다. 
이 예제는 간단한 언어 및 인덱스 매핑을 사용하고 있습니다.

실제 어플리케이션에서는 더욱 복잡한 어휘와 토크나이징 방법이 사용되야 함을 명심하세요.
e.g. BPE, SentencePiece, BBPE, LlamaTokenizer
"""


import torch
import numpy as np
from torch import nn

def get_device():
    """CUDA 사용 가능 여부에 따라 디바이스 설정"""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        """포지셔널 인코딩 생성"""
        # 포지션에 따른 각도 계산
        position = torch.arange(0, self.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_sequence_length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class SentenceEmbedding(nn.Module):
    """문장 임베딩을 생성하는 클래스"""
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.language_to_index = language_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        """배치에 대한 토크나이즈 작업 수행"""
        tokenized = []
        for sentence in batch:
            tokenized.append(self.tokenize(sentence, start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def tokenize(self, sentence, start_token=True, end_token=True):
        """문장을 토큰화하고 인덱스로 변환"""
        tokens = [self.language_to_index.get(token, self.language_to_index[self.PADDING_TOKEN]) for token in list(sentence)]
        if start_token:
            tokens.insert(0, self.language_to_index[self.START_TOKEN])
        if end_token:
            tokens.append(self.language_to_index[self.END_TOKEN])
        tokens += [self.language_to_index[self.PADDING_TOKEN]] * (self.max_sequence_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, x, start_token=True, end_token=True):
        """문장 임베딩 생성"""
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = x + pos
        x = self.dropout(x)
        return x

# 예제 사용법
if __name__ == "__main__":
    # 가상의 언어 및 인덱스 매핑
    language_to_index = {'<s>': 0, '</s>': 1, '<pad>': 2, 'H': 3, 'e': 4, 'l': 5, 'o': 6, ' ': 7, 'w': 8, 'r': 9, 'd': 10}
    max_sequence_length = 10
    d_model = 512

    # 모델 및 예제 데이터 준비
    model = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN='<s>', END_TOKEN='</s>', PADDING_TOKEN='<pad>')
    sentences = ["Hello world", "Hello world"]

    # 문장 임베딩 생성
    embeddings = model(sentences)
    print(embeddings.shape)  # 결과: torch.Size([2, 10, 512]) - 배치 크기: 2, 시퀀스 길이: 10, 임베딩 차원: 512
