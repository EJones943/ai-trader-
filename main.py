# ai_trading_ai.py

# ====== Imports ======
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
import uvicorn


# ====== Historical Data Utilities ======
def get_stock_data(ticker="AAPL", period="1y", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ====== Image Feature Extraction ======
class ChartFeatureExtractor:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model.eval()
        self.model.fc = nn.Identity()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_bytes):
        img = Image.open(image_bytes).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.squeeze().numpy()


# ====== AI Model ======
class TradingDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.numerical_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.visual_branch = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.combined = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # [buy/sell/hold, tp, sl]
        )

    def forward(self, numeric, visual):
        num_out = self.numerical_branch(numeric)
        vis_out = self.visual_branch(visual)
        combined = torch.cat((num_out, vis_out), dim=1)
        return self.combined(combined)


# ====== Dataset and Training (Optional) ======
class TradeDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, idx):
        num, vis, label = self.data[idx]
        return torch.tensor(num, dtype=torch.float32), \
               torch.tensor(vis, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

def train_model(model, dataset, epochs=20):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for numeric, visual, label in loader:
            output = model(numeric, visual)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# ====== FastAPI Backend ======
app = FastAPI()
model = TradingDecisionModel()
model.load_state_dict(torch.load("model.pt"))  # Pre-trained model
model.eval()
extractor = ChartFeatureExtractor()

@app.post("/predict/")
async def predict(file: UploadFile):
    visual_feat = extractor.extract(file.file)
    # Dummy numeric features - in practice, compute from current data
    numeric_feat = [100, 145.5, 0.75, 140, 148, 144]
    with torch.no_grad():
        result = model(
            torch.tensor([numeric_feat], dtype=torch.float32),
            torch.tensor([visual_feat], dtype=torch.float32)
        )
    action = ["Buy", "Sell", "Hold"][torch.argmax(result[0][:1]).item()]
    tp = result[0][1].item()
    sl = result[0][2].item()
    return {"action": action, "take_profit": tp, "stop_loss": sl}


# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=10000)
