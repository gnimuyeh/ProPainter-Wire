import os
import subprocess
import shutil
import time

# Import from refactored script (imports torch here, once)
from inference_propainter import load_models, run_inference, get_device

BAIDU_PCS = "/workspace/BaiduPCS-Go-v3.9.7-linux-amd64/BaiduPCS-Go"
REMOTE_FOLDER = "/我的资源/demo镜头"
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

total_start = time.time()

for file in all_files:
    if '_mask.' in file or '_result.' in file:
        continue

    basename, ext = os.path.splitext(file)
    mask_name = f"{basename}_mask{ext}"
    result_name = f"{basename}_result"

    input_path = os.path.join(LOCAL_INPUT_DIR, file)
    mask_path = os.path.join(LOCAL_INPUT_DIR, mask_name)
    binary_mask_path = os.path.join(LOCAL_INPUT_DIR, f"{basename}_mask_binary{ext}")
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

    video_start = time.time()

    # Download if not exists
    print(f"⬇️ Checking downloads for {file} and {mask_name}...")
    if os.path.exists(input_path):
        print(f"✅ Skipping download for {file} — already exists locally.")
    else:
        print(f"⬇️ Downloading {file}...")
        subprocess.run([BAIDU_PCS, "download", f"{REMOTE_FOLDER}/{file}", "--saveto", LOCAL_INPUT_DIR], check=True)

    if os.path.exists(mask_path):
        print(f"✅ Skipping download for {mask_name} — already exists locally.")
    else:
        print(f"⬇️ Downloading {mask_name}...")
        subprocess.run([BAIDU_PCS, "download", f"{REMOTE_FOLDER}/{mask_name}", "--saveto", LOCAL_INPUT_DIR], check=True)
        # --- Convert mask to binary ---
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", mask_path,
                "-vf", "format=gray,geq='if(gt(p(X,Y),128),255,0)'",
                "-pix_fmt", "gray", "-c:v", "ffv1", binary_mask_path
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        print(f"🧹 Converted {mask_name} to binary mask.")

    # Run inference (reuses models)
    print(f"🧠 Running ProPainter on {file}...")
    run_inference(
        video=input_path,
        mask=binary_mask_path,
        output=output_path,
        subvideo_length=10,
        raft_iter=30,
        ref_stride=10,
        mask_dilation=0,
        neighbor_length=10,
        fp16=False,  # Set to True if desired
        save_frames=False,
        save_masked_in=False,
        models=models
    )

    video_end = time.time()
    print(f"Processing time for {file}: {video_end - video_start:.2f} seconds")

    # Upload
    # print(f"📤 Uploading {result_name}...")
    # subprocess.run([BAIDU_PCS, "upload", "--nooverwrite=false", output_path, REMOTE_FOLDER], check=True)

    # # Cleanup
    # os.remove(input_path)
    # os.remove(mask_path)
    # os.remove(output_path)

total_end = time.time()
print(f"Total processing time: {total_end - total_start:.2f} seconds")

print("✅ All done!")
