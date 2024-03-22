import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer  # Transformer 모델 구현 파일

# 데이터셋 파일 경로 (영어와 한국어 데이터셋)
english_file = 'path/to/english_data.txt'  # 영어 데이터 파일 경로
korean_file = 'path/to/korean_data.txt'  # 한국어 데이터 파일 경로


# 어휘집 (간소화된 예시, 실제 어플리케이션에서는 더 많은 토큰 필요)
# START_TOKEN, PADDING_TOKEN, END_TOKEN은 각각 문장의 시작, 패딩, 종료를 나타내는 토큰입니다.
START_TOKEN = '<start>'
PADDING_TOKEN = '<pad>'
END_TOKEN = '<end>'

# 실제 어플리케이션에서는 더 많은 토큰이 포함되어야 합니다.
# Token (추후 개선: BBPE SentencePiece)
english_vocabulary = [START_TOKEN, 'Hello', 'world', '!',PADDING_TOKEN,END_TOKEN] 
korean_vocabulary = [START_TOKEN, '<pad>', '<end>', ' ', '안녕', '세상', '!',PADDING_TOKEN,END_TOKEN]


# 어휘집에서 토큰과 인덱스 사이의 매핑 생성
english_to_index = {token: index for index, token in enumerate(english_vocabulary)}
index_to_english = {index: token for index, token in enumerate(english_vocabulary)}
korean_to_index = {token: index for index, token in enumerate(korean_vocabulary)}
index_to_korean = {index: token for index, token in enumerate(korean_vocabulary)}


# 데이터셋 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, src_vocab, tgt_vocab):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        src_sentence = self.source_sentences[idx]
        tgt_sentence = self.target_sentences[idx]
        src_indices = [self.src_vocab[token] if token in self.src_vocab else self.src_vocab['<pad>'] for token in src_sentence.split(' ')]
        tgt_indices = [self.tgt_vocab[token] if token in self.tgt_vocab else self.tgt_vocab['<pad>'] for token in tgt_sentence.split(' ')]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


# 데이터셋 로딩 및 전처리
def load_and_preprocess_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]
    tgt_sentences = [line.strip() for line in tgt_sentences]
    return src_sentences, tgt_sentences

english_sentences, korean_sentences = load_and_preprocess_data(english_file, korean_file)
dataset = TranslationDataset(english_sentences, korean_sentences, english_to_index, korean_to_index)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Transformer 모델 정의
transformer = Transformer(
    d_model=512,
    ffn_hidden=2048,
    num_heads=8,
    drop_prob=0.1,
    num_layers=6,
    max_sequence_length=200,
    source_vocab_size=len(english_vocabulary),
    target_vocab_size=len(korean_vocabulary),
    target_pad_idx=korean_to_index['<pad>'],
    source_pad_idx=english_to_index['<pad>']
)


# 모델 훈련 로직
def train(model, data_loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=korean_to_index['<pad>'])

    for epoch in range(epochs):
        total_loss = 0
        for src_indices, tgt_indices in data_loader:
            optimizer.zero_grad()
            output = model(src_indices, tgt_indices[:, :-1])  # 마지막 <end> 토큰을 제외한 타겟 문장
            loss = criterion(output.view(-1, model.target_vocab_size), tgt_indices[:, 1:].reshape(-1))  # 첫 <start> 토큰을 제외한 타겟 문장
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")


# 평가 및 번역 로직
def translate(model, src_sentence, src_vocab, tgt_vocab, max_length=50):
    model.eval()
    src_indices = [src_vocab[token] if token in src_vocab else src_vocab['<pad>'] for token in src_sentence.split(' ')]
    src_tensor = torch.tensor(src_indices).unsqueeze(0)
    tgt_indices = [tgt_vocab['<start>']]
    
    for _ in range(max_length):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token_index = output.argmax(2)[:,-1].item()
        tgt_indices.append(next_token_index)
        if next_token_index == tgt_vocab['<end>']:
            break
    
    translated_sentence = ' '.join([index_to_korean[idx] for idx in tgt_indices[1:-1]])  # <start>와 <end> 토큰 제외
    return translated_sentence


# 모델 훈련
train(transformer, data_loader)


# 테스트 문장 번역
test_sentence = "Hello world !"
print("영어:", test_sentence)
print("번역된 한국어:", translate(transformer, test_sentence, english_to_index, korean_to_index))
