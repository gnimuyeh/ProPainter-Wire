FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Use the standard RunPod directory
WORKDIR /workspace

# --- 1. INSTALL SYSTEM DEPENDENCIES ---
# ProPainter / Video processing usually needs these for OpenCV/FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- 2. COPY SOURCE CODE ---
COPY . .

# Install directly into the system python (no Conda activation needed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- 3. RUN HANDLER ---
# Make sure the script is executable
RUN chmod +x runpod_handler.py
CMD ["python", "-u", "runpod_handler.py"]
