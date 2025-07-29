# Use RunPod's PyTorch base image with CUDA 12.8 and Ubuntu 22.04
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files into container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI server with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
