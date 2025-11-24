import argparse
import os
import stat
import subprocess
import shutil
import zipfile
import json
import math
import cv2  # Required for duration calculation
from inference_propainter import load_models, run_inference, get_device
from pathlib import Path

# Update signature to accept progress_callback
def run_job(job_id: str, source_url: str, progress_callback=None) -> dict:
    # ------------------- GLOBAL VARIABLES -------------------
    WORKSPACE = Path(__file__).resolve().parent
    LOCAL_INPUT_DIR   = os.path.join(WORKSPACE, "workdata", "input_videos")
    LOCAL_OUTPUT_DIR  = os.path.join(WORKSPACE, "workdata", "propainter_results")
    LOCAL_ZIP_PATH    = os.path.join(WORKSPACE, "workdata", "results.zip")
    REMOTE_JOB_PATH = f"/doya_jobs/{job_id}"
    REMOTE_ZIP_PATH = f"{REMOTE_JOB_PATH}/results.zip"
    
    # Force add execution rights
    BAIDU_PCS         = os.path.join(WORKSPACE, "BaiduPCS-Go")
    if os.path.exists(BAIDU_PCS):
        st = os.stat(BAIDU_PCS)
        os.chmod(BAIDU_PCS, st.st_mode | stat.S_IEXEC)

    # ------------------- Ensure Baidu login -------------------
    bduss = os.getenv("BAIDU_BDUSS")
    stoken = os.getenv("BAIDU_STOKEN")
    if not bduss or not stoken:
        raise RuntimeError("Missing BAIDU_BDUSS / BAIDU_STOKEN environment variables")
    subprocess.run([BAIDU_PCS, "login", f"-bduss={bduss}", f"-stoken={stoken}"], check=True)

    # ------------------- Clean & prepare local dirs -------------------
    for p in [LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    # ------------------- BaiduPCS-Go config & download -------------------
    if progress_callback: progress_callback("Initializing Download...")
    
    subprocess.run([BAIDU_PCS, "config", "set", "-savedir", LOCAL_INPUT_DIR], check=True)
    subprocess.run([BAIDU_PCS, "config", "set", "-max_parallel", "20"], check=True)
    subprocess.run([BAIDU_PCS, "mkdir", REMOTE_JOB_PATH], check=True)
    subprocess.run([BAIDU_PCS, "cd", REMOTE_JOB_PATH], check=True)
    
    # Handle URL splitting for password protection if needed
    if "?pwd=" in source_url:
        link, pwd = source_url.split("?pwd=")
        subprocess.run([BAIDU_PCS, "transfer", "--download", link, pwd], check=True)
    else:
        subprocess.run([BAIDU_PCS, "transfer", "--download", source_url], check=True)

    # ------------------- Find video directory -------------------
    video_dir = next(
        (root for root, _, files in os.walk(LOCAL_INPUT_DIR)
         if any(f.lower().endswith('.mov') for f in files)),
        None
    )
    if not video_dir:
        raise ValueError("No .mov files found after download")

    # ------------------- Count Files for Progress -------------------
    all_files = [f for f in os.listdir(video_dir) 
                 if f.lower().endswith('.mov') and '_mask.' not in f and '_result.' not in f]
    total_files = len(all_files)
    print(f"Found {total_files} videos to process", flush=True)

    # ------------------- Load models -------------------
    if progress_callback: progress_callback(f"Loading AI Models... (0/{total_files})")
    device = get_device()
    models = load_models(device, use_half=False)

    # ------------------- Process each video -------------------
    processed_count = 0
    total_seconds_accumulated = 0.0
    
    for file in all_files:
        basename, ext = os.path.splitext(file)
        video_path = os.path.join(video_dir, file)
        
        # --- PROGRESS UPDATE ---
        msg = f"Processing: {file} ({processed_count + 1}/{total_files})"
        print(msg, flush=True)
        if progress_callback: progress_callback(msg)
        # -----------------------

        mask_path = os.path.join(video_dir, f"{basename}_mask{ext}")
        if not os.path.exists(mask_path):
            print(f"Skipping {file} — no mask")
            # Cleanup skipped file to save space? Optional, but safer to leave if debugging.
            continue

        final_out = os.path.join(LOCAL_OUTPUT_DIR, f"{basename}_result{ext}")
        if os.path.exists(final_out):
            processed_count += 1
            continue

        # --- CALCULATE DURATION ---
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0:
                    duration = frame_count / fps
                    total_seconds_accumulated += duration
            cap.release()
        except Exception as e:
            print(f"Warning: Could not calculate duration for {file}: {e}")

        binary_mask = os.path.join(video_dir, f"{basename}_mask_binary{ext}")

        # Binary mask conversion
        subprocess.run([
            "ffmpeg", "-y", "-i", mask_path,
            "-vf", "format=gray,geq='if(gt(p(X,Y),128),255,0)'",
            "-pix_fmt", "gray", "-c:v", "ffv1", binary_mask
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Inference
        run_inference(
            video=video_path,
            mask=binary_mask,
            output=LOCAL_OUTPUT_DIR,
            subvideo_length=10,
            raft_iter=30,
            ref_stride=10,
            mask_dilation=0,
            neighbor_length=10,
            fp16=False,
            save_frames=False,
            save_masked_in=False,
            models=models
        )

        # Rename result
        temp_result = os.path.join(LOCAL_OUTPUT_DIR, "inpaint_out.mov")
        if os.path.exists(temp_result):
            shutil.move(temp_result, final_out)
        
        # --- CLEANUP INPUT FILES ---
        # Remove input video, input mask, and binary mask to save space
        try:
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(mask_path): os.remove(mask_path)
            if os.path.exists(binary_mask): os.remove(binary_mask)
        except Exception as e:
            print(f"Warning: Failed to cleanup files for {basename}: {e}")
        
        processed_count += 1

    # ------------------- Zip results -------------------
    if progress_callback: progress_callback("Zipping results...")
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(LOCAL_OUTPUT_DIR):
            for f in files:
                fp = os.path.join(root, f)
                arcname = os.path.relpath(fp, os.path.dirname(LOCAL_OUTPUT_DIR))
                z.write(fp, arcname)

    # ------------------- Upload & share -------------------
    if progress_callback: progress_callback("Uploading to Baidu Pan...")
    subprocess.run([BAIDU_PCS, "upload", LOCAL_ZIP_PATH, REMOTE_JOB_PATH], check=True)

    share = subprocess.run(
        [BAIDU_PCS, "share", "set", REMOTE_ZIP_PATH, "--period", "0"],
        capture_output=True, text=True, check=True
    )
    output = share.stdout.strip()
    parts = [p.strip() for p in output.split(',')]
    url = next((p.split('链接: ')[1] for p in parts if p.startswith('链接: ')), None)
    pwd = next((p.split('密码: ')[1] for p in parts if p.startswith('密码: ')), None)
    
    if not (url and pwd):
        raise RuntimeError(f"Failed to parse share link. Output: {output}")

    # Return structured data
    return {
        "url": f"{url}?pwd={pwd}",
        "duration": math.ceil(total_seconds_accumulated)
    }

# ==================== CLI ENTRYPOINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    try:
        result_data = run_job(args.job_id, args.source_url)
        print("✅ All done!")
        # Print JSON so the handler/UI can parse it cleanly
        print(json.dumps(result_data))
    except Exception as e:
        print("❌ Failed:", e)
        raise
