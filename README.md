# CV Project 1
 
이 프로젝트는 이미지 분류를 위한 PyTorch 기반의 머신러닝 파이프라인을 제공합니다. 데이터 로딩, 전처리, 모델 정의, 훈련, 검증, 추론의 모든 단계를 포함합니다.

## 디렉토리 구조

my_project/
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── loss.py
│   ├── trainer.py
│   └── inference.py
├── scripts/
│   ├── train.py
│   └── inference.py
├── requirements.txt
├── .gitignore
└── README.md

## 설치 방법

1. 가상 환경 생성 및 활성화:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 또는
    venv\Scripts\activate  # Windows
    ```

2. 필요한 패키지 설치:

    ```bash
    pip install -r requirements.txt
    ```

## 사용 방법

### 모델 훈련

```bash
python scripts/train.py
```

### 모델 추론

```bash
python scripts/inference.py
```

### 기여
기여는 언제나 환영합니다! 풀 리퀘스트를 보내주세요.