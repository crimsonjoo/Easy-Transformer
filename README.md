[![Transformer in Colab](https://img.shields.io/static/v1?label=Open%20in%20Colab&message=사용법&color=yellow&logo=googlecolab)](https://colab.research.google.com/drive/1OQy1F-xI53ft5yVX_qEvBEVT241R5Z3T?hl=ko#scrollTo=78fuZ01p83q5)


### 트랜스포머 모델 이해하기

트랜스포머 모델은 주로 자연어 처리(NLP) 분야에서 사용되며, 병렬 처리와 긴 거리의 의존성 학습에 강점을 가지고 있습니다. 이 모델은 인코더와 디코더라는 두 가지 주요 구성 요소로 이루어져 있습니다. 인코더는 입력 문장을 고차원적인 특징 벡터로 변환하고, 디코더는 이 특징 벡터를 사용하여 목표 언어로의 문장을 생성합니다.

## What are Transformers?

Transformers are a class of models that revolutionized the field of natural language processing (NLP). Introduced in the paper "Attention Is All You Need" by Vaswani et al., Transformers have set new benchmarks in a variety of tasks, including but not limited to translation, text generation, and sentiment analysis. Their unique architecture allows for parallel processing of sequences, making them significantly faster and more efficient than their predecessors.

## 구조 및 파일 설명

- `Encoder.py`: 트랜스포머 모델의 인코더 구성을 담당하는 파일입니다.
- `Decoder.py`: 트랜스포머 모델의 디코더 구성을 담당하는 파일입니다.
- `Transformer_Model.py`: 전체 트랜스포머 모델을 구성하는 파일입니다.
- `Transformer_translator_enkn.py`: 인코더-디코더를 모두 사용하는 트랜스포머 모델 번역기(영어-한국어) 예제입니다.
- `Transformer_translator_enkr.py`: 인코더-디코더를 모두 사용하는 트랜스포머 모델 번역기(영어-칸나다어) 예제입니다.

### 시작하기 전에

이 프로젝트를 시작하기 전에, Python과 기본적인 머신러닝 지식이 필요합니다. 특히, PyTorch 같은 딥러닝 프레임워크에 대한 경험이 있으면 좋습니다.

### 설치 방법

```
git clone https://github.com/crimsonjoo/Easy-Transformer.git
cd Easy-Transformer
```


