FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights so cold starts only pay network cost once
RUN python -c "\
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True); \
print('Model downloaded successfully')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
