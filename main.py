# ai_trading_ai.py

# ======= Imports =======
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import resnet50
import io

# ======= FastAPI App =======
app = FastAPI()

# ======= Global State =======
trained_model = None

# ======= Utility Functions =======
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(ticker="AAPL", period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# ======= ML Model Definition =======
class SimpleStockModel(nn.Module):
    def __init__(self, input_size=2):
        super(SimpleStockModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# ======= Training Function =======
def train_model(df):
    X = df[['RSI', 'MA20']].values
    y = df['Close'].shift(-1).fillna(method='ffill').values  # Next day's price

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = SimpleStockModel(input_size=2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()

    return model

# ======= Image Model for Chart Predictions =======
image_model = resnet50(pretrained=True)
image_model.eval()
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ======= FastAPI Routes =======

@app.get("/")
def root():
    return {"message": "AI stock trading server is running."}

@app.get("/stock/{ticker}")
def fetch_stock(ticker: str):
    try:
        df = get_stock_data(ticker)
        return df.tail(5).to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/train/{ticker}")
def train(ticker: str):
    global trained_model
    try:
        df = get_stock_data(ticker)
        trained_model = train_model(df)
        return {"message": f"Model trained on {ticker} successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class Features(BaseModel):
    rsi: float
    ma20: float

@app.post("/predict_price")
def predict_price(features: Features):
    global trained_model
    if trained_model is None:
        return JSONResponse(status_code=400, content={"error": "Model is not trained yet. Call /train/{ticker} first."})
    with torch.no_grad():
        x = torch.tensor([[features.rsi, features.ma20]], dtype=torch.float32)
        prediction = trained_model(x).item()
    return {"predicted_close": prediction}

@app.post("/predict_chart")
async def predict_chart(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = image_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        return {"chart_prediction": pred}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ======= Dev Server Entry Point =======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)

