import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch, torch.nn as nn
from pathlib import Path

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7,128), nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self, x): return self.net(x)

def load_labels():
    p = Path("artifacts/labels.txt")
    if p.exists(): return p.read_text().splitlines()
    return [str(i) for i in range(10)]

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = load_labels()
model = TinyCNN().to(device)
ckpt = torch.load("artifacts/model.pt", map_location=device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

app = FastAPI(title="FashionMNIST API")

@app.get("/health")
def health():
    return {"status":"ok","device":device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28,28))
    x = torch.tensor((255 - torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).float()/255.0)
    x = x.view(1,1,28,28).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=1).item())
    return JSONResponse({"label": labels[idx], "index": idx, "probs": probs})
