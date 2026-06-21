FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw models outputs logs

# 컨테이너 실행 시 모델이 없으면 학습 후 샘플 추론까지 수행
CMD ["bash", "-c", "python src/train.py && python src/inference.py --input data/sample_input.csv --output outputs/predictions.csv && cat outputs/predictions.csv"]
