from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config import settings

TARGET_LABELS = ["foggy", "blurry", "dark", "damaged", "normal"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuroVisionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(NeuroVisionModel, self).__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(weights_path, device):
    model = NeuroVisionModel(num_classes=len(TARGET_LABELS))

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    print("Damage Classifier Model loaded")

    model.to(device)
    model.eval()

    return model

@lru_cache(maxsize=1)
def get_classifier():
    """Shared lazy singleton — the classifier is loaded once per process."""
    return get_model(str(settings.models_dir / "classifier.pth"), DEVICE)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def smart_route(result):
    # Smart Routing Inference Logic
    # We use a lower threshold for degradations to be more sensitive to real-world defects.
    # The training data was heavily degraded, so the model might be under-confident on subtle defects.
    
    THRESH_DEGRADE = 0.3  # Lower threshold for degradations
    THRESH_NORMAL = 0.5   # Standard threshold for normal

    # checking for specific degradations with sensitive threshold
    detected_degradations = [
        label for label in ("foggy", "blurry", "dark", "damaged")
        if result["probs"][label] > THRESH_DEGRADE
    ]

    # If ANY degradation is detected, we route there, ignoring "normal" score
    if detected_degradations:
        result["route"].extend(detected_degradations)

    # Only if NO degradations are found do we check for Normal
    elif result["probs"]["normal"] > THRESH_NORMAL:
        result["route"].append("normal")
    
    else:
        # Fallback for weak signals
        # We find the max probability class that isn't normal (if normal is low)
        # or just state uncertain.
        result["route"].append("uncertain")

    return result

def predict_image(model, image_bytes, device=None):
    """Runs inference on a single image (PIL image or byte stream)."""

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    transform = get_transform()

    if isinstance(image_bytes, Image.Image):
        image = image_bytes.convert("RGB")
    else:
        image = Image.open(image_bytes).convert("RGB")
    
    # Preprocess
    img_t = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.sigmoid(logits)
    
    # Process results
    logits_np = logits.cpu().numpy()[0]
    probs_np = probs.cpu().numpy()[0]
    
    result = {
        "logits": {label: float(logits_np[i]) for i, label in enumerate(TARGET_LABELS)},
        "probs": {label: float(probs_np[i]) for i, label in enumerate(TARGET_LABELS)},
        "route": []
    }

    result = smart_route(result)

    return result
