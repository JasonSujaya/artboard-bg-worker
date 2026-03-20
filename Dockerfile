FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Link to public repo so GHCR inherits public visibility
LABEL org.opencontainers.image.source=https://github.com/JasonSujaya/artboard-bg-worker

WORKDIR /app

# Install dependencies (torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights (BiRefNet is open / ungated)
RUN python -c "\
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True); \
print('Model downloaded successfully')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
