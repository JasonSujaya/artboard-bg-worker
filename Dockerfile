FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install dependencies (torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights so cold starts are fast
RUN python -c "\
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True); \
print('Model downloaded successfully')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
