"""
RunPod Serverless Handler — RMBG-2.0 Background Removal

Accepts a base64-encoded image, returns:
  - alpha_mask: base64 PNG (grayscale mask)
  - foreground: base64 RGBA PNG (cutout with transparency)
  - width / height of the original image
"""

import base64
import io

import numpy as np
import runpod
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

# ---------------------------------------------------------------------------
# Model loading (runs once at worker startup, outside the handler)
# ---------------------------------------------------------------------------
print("[bg-removal] Loading RMBG-2.0 model…")
from transformers import AutoModelForImageSegmentation  # noqa: E402

MODEL_INPUT_SIZE = (1024, 1024)

model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"[bg-removal] Model loaded on {device}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to normalised model-ready tensor."""
    arr = np.array(image)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    tensor = F.interpolate(tensor.unsqueeze(0), size=MODEL_INPUT_SIZE, mode="bilinear")
    tensor = torch.divide(tensor, 255.0)
    tensor = normalize(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return tensor


def postprocess(pred: torch.Tensor, orig_size: tuple[int, int]) -> np.ndarray:
    """Resize prediction back to original dimensions and convert to uint8."""
    result = torch.squeeze(
        F.interpolate(pred, size=orig_size, mode="bilinear"), 0
    )
    ma, mi = torch.max(result), torch.min(result)
    result = (result - mi) / (ma - mi)
    arr = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    return np.squeeze(arr)


def encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    inp = event.get("input", {})
    image_b64 = inp.get("image_base64")

    if not image_b64:
        return {"error": "Missing 'image_base64' in input"}

    try:
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return {"error": f"Failed to decode image: {exc}"}

    orig_w, orig_h = image.size

    # Inference
    input_tensor = preprocess(image).to(device)
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()

    mask_arr = postprocess(preds, (orig_h, orig_w))
    mask_img = Image.fromarray(mask_arr, mode="L")

    # Build RGBA foreground
    rgba = image.copy().convert("RGBA")
    rgba.putalpha(mask_img)

    return {
        "alpha_mask": encode_png(mask_img),
        "foreground": encode_png(rgba),
        "width": orig_w,
        "height": orig_h,
    }


runpod.serverless.start({"handler": handler})
