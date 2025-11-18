FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /runpod-volume/ProPainter-Wire

CMD ["/bin/bash", "-c", "\
    export PATH=\"/runpod-volume/miniconda3/bin:$PATH\" && \
    conda activate /runpod-volume/envs/propainter && \
    exec python -u runpod_handler.py"]
