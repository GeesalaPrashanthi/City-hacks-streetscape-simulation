"""
Run trained classifier on a sidewalk image.
Usage: python predict.py <image_path_or_url>
"""
import os
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMG_SIZE_DEFAULT = 224


def resolve_model_path():
    env_model = os.getenv("MODEL_PATH", "").strip()
    if env_model:
        return Path(env_model)

    for candidate in ["sidewalk_classifier_fair.pt", "sidewalk_classifier.pt"]:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return Path("sidewalk_classifier.pt")


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(arch: str, num_classes: int):
    if arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported checkpoint architecture: {arch}")


MODEL_PATH = resolve_model_path()
CHECKPOINT = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = CHECKPOINT["classes"]
ARCH = CHECKPOINT.get("arch", "efficientnet_b2")
IMG_SIZE = int(CHECKPOINT.get("img_size", IMG_SIZE_DEFAULT))
VAL_SCORE = CHECKPOINT.get("val_balanced_acc", CHECKPOINT.get("val_acc", None))
TF = make_transform(IMG_SIZE)

MODEL = build_model(ARCH, len(CLASSES))
MODEL.load_state_dict(CHECKPOINT["model_state"])
MODEL.eval()

if VAL_SCORE is None:
    print(f"Loaded model: {MODEL_PATH} | Arch: {ARCH} | Classes: {CLASSES}")
else:
    print(f"Loaded model: {MODEL_PATH} | Arch: {ARCH} | Classes: {CLASSES} | ValScore: {VAL_SCORE:.3f}")


def predict(image_source: str):
    if image_source.startswith("http"):
        with urllib.request.urlopen(image_source) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    tensor = TF(image).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    result = {cls: round(float(probs[i]), 4) for i, cls in enumerate(CLASSES)}
    predicted = max(result, key=result.get)
    print(f"Predicted: {predicted}")
    print(f"Probabilities: {result}")
    return predicted, result


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else ""
    if src:
        predict(src)
