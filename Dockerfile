FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set working directory to your project folder on the volume
WORKDIR /workspace/ProPainter-Wire

CMD ["/bin/bash", "-c", "\
    export PATH=\"/workspace/miniconda3/bin:$PATH\" && \
    source /workspace/miniconda3/bin/activate /workspace/envs/propainter && \
    exec python -u runpod_handler.py"]
