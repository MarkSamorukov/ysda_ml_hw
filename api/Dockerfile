FROM python:3.9-slim

WORKDIR /app
COPY ./app /app

RUN pip install --no-cache-dir \
    torch \
    transformers \
    fastapi \
    uvicorn \
    numpy \
    scikit-learn

CMD ["python", "main.py"]