# CardioCare: End-to-End Machine Learning System

## 사용 데이터셋

본 프로젝트는 업로드된 UCI Heart Disease 원본 패키지 중 processed 데이터 4개를 사용합니다.

```text
data/raw/processed.cleveland.data
data/raw/processed.hungarian.data
data/raw/processed.switzerland.data
data/raw/processed.va.data
```

각 파일은 공식 14개 컬럼을 사용합니다.

```text
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, num
```

타깃 `num`은 `num == 0` 정상, `num > 0` 심장병 있음으로 이진화합니다. 여러 병원 데이터 출처를 합친 뒤 `source` 컬럼은 데이터 출처 편향을 줄이기 위해 학습 특성에서 제거합니다.

심장병 발병 가능성을 예측하여 심장 전문의의 의사결정을 보조하는 End-to-End ML 프로젝트입니다.

> 윤리 원칙: CardioCare는 의사를 대체하지 않습니다. 모델은 정보를 제공하고(inform), 최종 의사결정은 전문의가 수행합니다.

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 학습 실행

```bash
python src/train.py
```

실행 후 생성 파일:

```text
models/model.joblib
models/model_metadata.json
outputs/model_comparison.csv
outputs/final_confusion_matrix.png
outputs/selected_features.txt
mlruns/
```

## 3. 테스트 실행

```bash
python -m unittest
```

## 4. 추론 실행

```bash
python src/inference.py --input data/sample_input.csv --output outputs/predictions.csv
```

## 5. 모니터링 및 드리프트 실행

```bash
python src/monitor.py
```

생성 파일:

```text
logs/predictions.log
outputs/drift_ks_report.csv
outputs/drift_performance_report.csv
outputs/metric_timeseries.png
```

## 6. Docker 빌드 및 실행

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

## 7. MLflow 확인

```bash
mlflow ui
```

브라우저에서 `http://127.0.0.1:5000` 접속 후 3개 이상의 모델 실행 기록을 확인합니다.

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/
│   └── sample_input.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── models/
├── outputs/
├── logs/
├── Dockerfile
├── requirements.txt
├── README.md
└── .github/workflows/ci.yml
```

## AI 도구 사용 공개

본 프로젝트의 보일러플레이트 코드 작성과 디버깅 보조에 ChatGPT를 사용했습니다. 최종 코드 실행, 결과 해석, 모델 선택 및 보고서 내용에 대한 책임은 제출자 본인에게 있습니다.
