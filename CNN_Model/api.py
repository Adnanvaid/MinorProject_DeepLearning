from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import CNN   # your CNN architecture

app = FastAPI()

# Load trained model
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# CIFAR10 Classes
classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# Image preprocessing (like scaling in tabular model)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    image = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {
        "predicted_class": classes[predicted.item()]
    }