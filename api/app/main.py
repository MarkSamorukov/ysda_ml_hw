from fastapi import FastAPI
from predict import predict_with_confidence, load_models
import uvicorn

app = FastAPI()

# Загружаем модель при старте
@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/predict")
async def predict(title: str, abstract: str = ""):
    return predict_with_confidence(title, abstract)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)