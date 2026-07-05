import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.dark.classes import RAGRetinexFormer
from config import settings
from RAG.similarity_search import find_similar_images

MODEL_WEIGHTS_PATH = str(settings.models_dir / "retinex_gan_8.pth")

IMAGE_SIZE = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

resize_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

model = RAGRetinexFormer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()


def load_reference_tensors(query_image, k=3):
    """Retrieve k well-lit reference images for the query (PIL image or path)."""
    ref_tensors = []
    try:
        for path, dataset, dist in find_similar_images(query_image, k=k):
            img = Image.open(path).convert("RGB")
            ref_tensors.append(resize_transform(img))
            print(f"Retrieved: {path} | {dataset} | {dist}")
    except Exception as e:
        print(f"Similarity search failed: {e}")

    if not ref_tensors:
        return torch.zeros(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    return torch.stack(ref_tensors).unsqueeze(0).to(DEVICE)


def de_dark(pil_image, report=None):
    """Low-light enhancement with RAG similarity references."""
    input_tensor = input_transform(pil_image).unsqueeze(0).to(DEVICE)
    ref_tensors = load_reference_tensors(pil_image, k=3)

    with torch.no_grad():
        output = model(input_tensor, ref_tensors)

    enhanced = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    enhanced_uint8 = ((enhanced * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    denoised_rgb = cv2.fastNlMeansDenoisingColored(
        enhanced_uint8,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return Image.fromarray(denoised_rgb)
