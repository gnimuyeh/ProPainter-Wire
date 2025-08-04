import os
import subprocess
import shutil
import time
# Import from refactored script (imports torch here, once)
from inference_propainter import load_models, run_inference, get_device
LOCAL_INPUT_DIR = "/workspace/workdata/input_videos"
LOCAL_OUTPUT_DIR = "/workspace/workdata/propainter_results"
os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
# Load models once
device = get_device()
models = load_models(device, use_half=False) # Set use_half=True if desired; fp16=False by default
# Get file list locally
print(f"📁 Listing files in {LOCAL_INPUT_DIR}...")
all_files = [f for f in os.listdir(LOCAL_INPUT_DIR) if f.endswith('.mov')]
total_start = time.time()
for file in all_files:
    if '_mask.' in file or '_result.' in file:
        continue
    basename, ext = os.path.splitext(file)
    mask_name = f"{basename}_mask{ext}"
    result_name = f"{basename}_result{ext}"
    result_name_check = f"{basename}_pnt_v002{ext}"
    input_path = os.path.join(LOCAL_INPUT_DIR, file)
    mask_path = os.path.join(LOCAL_INPUT_DIR, mask_name)
    output_path = os.path.join(LOCAL_OUTPUT_DIR, result_name)
    output_path_check = os.path.join(LOCAL_OUTPUT_DIR, result_name_check)
    if os.path.exists(output_path_check) or os.path.exists(output_path):
        print(f"✅ Skipping {file} — local result exists.")
        continue
    # Check mask
    if not os.path.exists(mask_path):
        print(f"❌ Skipping {file} — no mask.")
        continue
    video_start = time.time()
    # Run inference (reuses models)
    print(f"🧠 Running ProPainter on {file}...")
    run_inference(
        video=input_path,
        mask=mask_path,
        output=output_path,
        subvideo_length=10,
        raft_iter=50,
        ref_stride=5,
        mask_dilation=0,
        neighbor_length=10,
        fp16=False,  # Set to True if desired
        save_masked_in=False,
        models=models
    )
    video_end = time.time()
    print(f"Processing time for {file}: {video_end - video_start:.2f} seconds")
total_end = time.time()
print(f"Total processing time: {total_end - total_start:.2f} seconds")
print("✅ All done!")
