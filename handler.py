"""
RunPod Serverless Handler — BiRefNet Background Removal
"""

import base64
import io
import traceback

import runpod

# Deferred model loading — load on first request to avoid startup timeout
model = None
device = None

def load_model():
    global model, device
    if model is not None:
        return

    print("[bg-removal] Loading BiRefNet model…")
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[bg-removal] Model loaded on {device}")


def handler(event: dict) -> dict:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms.functional import normalize

        load_model()

        inp = event.get("input", {})
        image_b64 = inp.get("image_base64")

        if not image_b64:
            return {"error": "Missing 'image_base64' in input"}

        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_w, orig_h = image.size

        # Preprocess
        arr = np.array(image)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear")
        tensor = torch.divide(tensor, 255.0)
        tensor = normalize(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        input_tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid()

        # Postprocess
        result = torch.squeeze(F.interpolate(preds, size=(orig_h, orig_w), mode="bilinear"), 0)
        ma, mi = torch.max(result), torch.min(result)
        result = (result - mi) / (ma - mi)
        mask_arr = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        mask_arr = np.squeeze(mask_arr)

        mask_img = Image.fromarray(mask_arr, mode="L")
        rgba = image.copy().convert("RGBA")
        rgba.putalpha(mask_img)

        # Encode
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format="PNG", optimize=True)
        fg_buf = io.BytesIO()
        rgba.save(fg_buf, format="PNG", optimize=True)

        return {
            "alpha_mask": base64.b64encode(mask_buf.getvalue()).decode(),
            "foreground": base64.b64encode(fg_buf.getvalue()).decode(),
            "width": orig_w,
            "height": orig_h,
        }
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}


print("[bg-removal] Starting RunPod serverless worker…")
runpod.serverless.start({"handler": handler})
