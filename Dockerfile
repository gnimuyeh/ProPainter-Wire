FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# CMD ["/bin/bash", "-c", "\
#     export PATH=\"/workspace/miniconda3/bin:$PATH\" && \
#     source /workspace/miniconda3/bin/activate /workspace/envs/propainter && \
#     exec python -u /workspace/ProPainter-Wire/runpod_handler.py"]
    

# ← THIS LINE DROPS YOU INTO A SHELL SO YOU CAN LOOK AROUND
CMD ["sleep", "infinity"]
