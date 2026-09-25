FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONHASHSEED=42
ENV TORCH_DETERMINISTIC=1
RUN python3 frontier-01-usual-attention-code.py && python3 frontier-01-high-level-demo.py && python3 frontier-01-eval-high-level.py
CMD ["python3", "frontier-01-eval-high-level.py"]
