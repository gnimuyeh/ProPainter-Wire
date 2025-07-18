import os
import subprocess
import shutil

# Import from refactored script (imports torch here, once)
from inference_propainter import load_models, run_inference, get_device

BAIDU_PCS = "/workspace/BaiduPCS-Go-v3.9.7-linux-amd64/BaiduPCS-Go"
REMOTE_FOLDER = "/我的资源/豆芽威亚alpha"
LOCAL_INPUT_DIR = "/workspace/workdata/input_videos"
LOCAL_OUTPUT_DIR = "/workspace/workdata/propainter_results"

os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Load models once
device = get_device()
models = load_models(device, use_half=False)  # Set use_half=True if desired; fp16=False by default

# Get file list once
print(f"📁 Listing files in {REMOTE_FOLDER}...")
result = subprocess.run([BAIDU_PCS, "ls", REMOTE_FOLDER], capture_output=True, text=True, check=True)
all_files = [line.split()[-1] for line in result.stdout.splitlines() if '.mov' in line]

for file in all_files:
    if '_mask.' in file or '_result.' in file:
        continue

    basename, ext = os.path.splitext(file)
    mask_name = f"{basename}_mask{ext}"
    result_name = f"{basename}_result{ext}"

    input_path = os.path.join(LOCAL_INPUT_DIR, file)
    mask_path = os.path.join(LOCAL_INPUT_DIR, mask_name)
    output_path = os.path.join(LOCAL_OUTPUT_DIR, result_name)

    if os.path.exists(output_path):
        print(f"✅ Skipping {file} — local result exists.")
        continue

    # Check remote result (using cached all_files)
    if result_name in all_files:
        print(f"✅ Skipping {file} — result exists.")
        continue

    # Check mask (using cached all_files)
    if mask_name not in all_files:
        print(f"❌ Skipping {file} — no mask.")
        continue

    # Download
    print(f"⬇️ Downloading {file} and {mask_name}...")
    subprocess.run([BAIDU_PCS, "download", f"{REMOTE_FOLDER}/{file}", "--saveto", LOCAL_INPUT_DIR], check=True)
    subprocess.run([BAIDU_PCS, "download", f"{REMOTE_FOLDER}/{mask_name}", "--saveto", LOCAL_INPUT_DIR], check=True)

    # Run inference (reuses models)
    print(f"🧠 Running ProPainter on {file}...")
    run_inference(
        video=input_path,
        mask=mask_path,
        output=output_path,
        subvideo_length=10,
        raft_iter=50,
        ref_stride=10,
        mask_dilation=0,
        neighbor_length=10,
        fp16=False,  # Set to True if desired
        save_masked_in=False,
        models=models
    )

    # # Upload
    # print(f"📤 Uploading {result_name}...")
    # subprocess.run([BAIDU_PCS, "upload", output_path, REMOTE_FOLDER], check=True)

    # # Cleanup
    # os.remove(input_path)
    # os.remove(mask_path)
    # os.remove(output_path)

print("✅ All done!")
