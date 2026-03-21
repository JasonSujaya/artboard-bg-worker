"""
RunPod Serverless Handler — Image Tools

Supports two operations via the "action" field:
  - "remove_bg": BiRefNet background removal
  - "upscale": Real-ESRGAN 4x upscaling
"""

import base64
import io
import traceback

import runpod

# Patch: basicsr imports removed torchvision.transforms.functional_tensor
import torchvision.transforms.functional as _F
import torchvision.transforms
torchvision.transforms.functional_tensor = _F

# ---------------------------------------------------------------------------
# Lazy model caches
# ---------------------------------------------------------------------------
birefnet_model = None
upscaler_models = {}
device = None


def get_device():
    global device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_birefnet():
    global birefnet_model
    if birefnet_model is not None:
        return birefnet_model

    print("[tools] Loading BiRefNet…")
    import torch
    from transformers import AutoModelForImageSegmentation

    birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet_model.to(get_device())
    birefnet_model.eval()
    birefnet_model.float()
    print(f"[tools] BiRefNet loaded on {get_device()}")
    return birefnet_model


def load_upscaler(model_name: str = "RealESRGAN_x4plus_anime_6B", scale: int = 4):
    key = f"{model_name}_{scale}"
    if key in upscaler_models:
        return upscaler_models[key]

    print(f"[tools] Loading {model_name} (scale={scale})…")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Model configs
    configs = {
        "RealESRGAN_x4plus": {
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
        },
        "RealESRGAN_x2plus": {
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
        },
        "4x_Remacri": {
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_foolhardy_Remacri.pth",
            "scale": 4,
        },
        "4x_UltraSharp": {
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_UltraSharp.pth",
            "scale": 4,
        },
    }

    cfg = configs.get(model_name, configs["RealESRGAN_x4plus_anime_6B"])
    upsampler = RealESRGANer(
        scale=cfg["scale"],
        model_path=cfg["url"],
        model=cfg["arch"](),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0 if get_device() == "cuda" else None,
    )
    upscaler_models[key] = upsampler
    print(f"[tools] {model_name} loaded")
    return upsampler


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def do_remove_bg(inp: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import normalize

    model = load_birefnet()
    image_b64 = inp.get("image_base64", "")
    if not image_b64:
        return {"error": "Missing 'image_base64'"}

    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_w, orig_h = image.size

    arr = np.array(image)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear")
    tensor = torch.divide(tensor, 255.0)
    tensor = normalize(tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = model(tensor.to(get_device()))[-1].sigmoid()

    result = torch.squeeze(F.interpolate(preds, size=(orig_h, orig_w), mode="bilinear"), 0)
    ma, mi = torch.max(result), torch.min(result)
    result = (result - mi) / (ma - mi)
    mask_arr = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    mask_arr = np.squeeze(mask_arr)

    mask_img = Image.fromarray(mask_arr, mode="L")
    rgba = image.copy().convert("RGBA")
    rgba.putalpha(mask_img)

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


def do_upscale(inp: dict) -> dict:
    import cv2
    import numpy as np
    from PIL import Image

    image_b64 = inp.get("image_base64", "")
    if not image_b64:
        return {"error": "Missing 'image_base64'"}

    model_name = inp.get("model", "RealESRGAN_x4plus_anime_6B")
    scale = inp.get("scale", 4)

    upsampler = load_upscaler(model_name, scale)

    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    output, _ = upsampler.enhance(img_cv, outscale=scale)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(output_rgb)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "width": out_img.width,
        "height": out_img.height,
        "model": model_name,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    try:
        inp = event.get("input", {})
        action = inp.get("action", "remove_bg")

        if action == "remove_bg":
            return do_remove_bg(inp)
        elif action == "upscale":
            return do_upscale(inp)
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}


print("[tools] Starting RunPod serverless worker…")
runpod.serverless.start({"handler": handler})
