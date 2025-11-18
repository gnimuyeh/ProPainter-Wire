FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

CMD ["/bin/bash", "-c", "\
    export PATH=\"/runpod-volume/miniconda3/bin:$PATH\" && \
    source /runpod-volume/miniconda3/bin/activate /runpod-volume/envs/propainter && \
    exec python -u /runpod-volume/ProPainter-Wire/runpod_handler.py"]
