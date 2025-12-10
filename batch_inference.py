import argparse
import os
import stat
import subprocess
import shutil
import zipfile
import json
import math
import cv2
import torch 
import gc
from inference_propainter import load_models, run_inference, get_device
from pathlib import Path

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

    # ------------------- Prepare local dirs -------------------
    os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # ------------------- BaiduPCS-Go Download -------------------
    if progress_callback: progress_callback("Initializing Download...")
    
    subprocess.run([BAIDU_PCS, "config", "set", "-savedir", LOCAL_INPUT_DIR], check=True)
    subprocess.run([BAIDU_PCS, "config", "set", "-max_parallel", "20"], check=True)
    # Try creating remote dir (ignore if exists)
    subprocess.run([BAIDU_PCS, "mkdir", REMOTE_JOB_PATH], check=False, stdout=subprocess.DEVNULL) 
    subprocess.run([BAIDU_PCS, "cd", REMOTE_JOB_PATH], check=True)
    
    try:
        if "?pwd=" in source_url:
            link, pwd = source_url.split("?pwd=")
            subprocess.run([BAIDU_PCS, "transfer", "--download", link, pwd], check=True)
        else:
            subprocess.run([BAIDU_PCS, "transfer", "--download", source_url], check=True)
    except subprocess.CalledProcessError as e:
        msg = f"⚠️ Download error {e.returncode}. Checking cache..."
        print(msg, flush=True)

    # ------------------- Find video directory -------------------
    video_dir = next(
        (root for root, _, files in os.walk(LOCAL_INPUT_DIR)
         if any(f.lower().endswith('.mov') for f in files)),
        None
    )
    
    if not video_dir:
        raise ValueError("No .mov files found (Download failed and no local cache)")

    # ------------------- Filter Files -------------------
    # Improved filtering: Exclude results, masks, and hidden files
    all_files = [f for f in os.listdir(video_dir) 
                 if f.lower().endswith('.mov') 
                 and '_mask' not in f 
                 and '_result' not in f
                 and not f.startswith('.')]
    
    total_files = len(all_files)
    print(f"Found {total_files} videos to process", flush=True)

    # ------------------- Load models -------------------
    if progress_callback: progress_callback(f"Loading AI Models... (0/{total_files})")
    device = get_device()
    models = load_models(device, use_half=True)

    # ------------------- Process Loop -------------------
    processed_count = 0
    total_seconds_accumulated = 0.0
    failed_files = [] 
    
    for file in all_files:
        basename, ext = os.path.splitext(file)
        video_path = os.path.join(video_dir, file)
        
        # Look for mask with matching extension
        mask_path = os.path.join(video_dir, f"{basename}_mask{ext}")
        
        # Fallback: Check for .mp4 mask if .mov doesn't exist (common in masking workflows)
        if not os.path.exists(mask_path):
             mask_path_alt = os.path.join(video_dir, f"{basename}_mask.mp4")
             if os.path.exists(mask_path_alt):
                 mask_path = mask_path_alt

        final_out = os.path.join(LOCAL_OUTPUT_DIR, f"{basename}_result{ext}")

        # --- PROGRESS UPDATE ---
        msg = f"Processing: {file} ({processed_count + 1}/{total_files})"
        print(msg, flush=True)
        if progress_callback: progress_callback(msg)

        try:
            # 1. Validation Checks
            if os.path.exists(final_out):
                print(f"Skipping {file} — result exists")
                processed_count += 1
                continue 

            if not os.path.exists(mask_path):
                print(f"Skipping {file} — no mask found")
                continue

            # 2. Duration Calculation (Quickly)
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps_val = cap.get(cv2.CAP_PROP_FPS)
                    f_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps_val > 0:
                        total_seconds_accumulated += (f_count / fps_val)
                cap.release()
            except: pass

            # 3. Inference (Directly passing mask_path, NO FFmpeg conversion step)
            run_inference(
                video=video_path,
                mask=mask_path,  # Passing the raw mask video path
                output=LOCAL_OUTPUT_DIR,
                subvideo_length=30, # Increased for better temporal consistency if VRAM allows
                raft_iter=20,       # 20 is usually sufficient for wires, 30 is safer
                ref_stride=10,
                mask_dilation=0,    # Slight dilation handles compression artifacts in mask
                neighbor_length=10,
                fp16=True,
                save_frames=False,
                save_masked_in=False,
                models=models,
                device=device
            )

            # 4. Rename result (Inference saves as 'inpaint_out.mov')
            temp_result = os.path.join(LOCAL_OUTPUT_DIR, "inpaint_out.mov")
            if os.path.exists(temp_result):
                shutil.move(temp_result, final_out)
            
            processed_count += 1
            
            # --- CRITICAL: VRAM CLEANUP ---
            # Python's GC doesn't always free Torch tensors immediately
            gc.collect()
            torch.cuda.empty_cache()
            # ------------------------------

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error processing {file}: {error_msg}")
            failed_files.append({"file": file, "error": error_msg})
            
            # Clean up potential partial file
            if os.path.exists(os.path.join(LOCAL_OUTPUT_DIR, "inpaint_out.mov")):
                os.remove(os.path.join(LOCAL_OUTPUT_DIR, "inpaint_out.mov"))
                
            # Try to reset memory even on failure
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------- Zip results -------------------
    has_results = any(os.scandir(LOCAL_OUTPUT_DIR))
    
    if not has_results and failed_files:
        raise RuntimeError(f"All {total_files} videos failed. Errors: {failed_files}")

    if progress_callback: progress_callback("Zipping results...")
    
    # Improved Zipping: Store relative paths correctly to avoid junk folders
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(LOCAL_OUTPUT_DIR):
            for f in files:
                if not f.startswith('.'):
                    fp = os.path.join(root, f)
                    z.write(fp, f) # Write file at root of zip

    # ------------------- Upload & share -------------------
    if progress_callback: progress_callback("Uploading to Baidu Pan...")
    subprocess.run([BAIDU_PCS, "upload", LOCAL_ZIP_PATH, REMOTE_JOB_PATH], check=True)

    share = subprocess.run(
        [BAIDU_PCS, "share", "set", REMOTE_ZIP_PATH, "--period", "0"],
        capture_output=True, text=True, check=True
    )
    output = share.stdout.strip()
    
    # Robust parsing
    url = None; pwd = None
    if '链接: ' in output: url = output.split('链接: ')[1].split(' ')[0].strip()
    if '密码: ' in output: pwd = output.split('密码: ')[1].split(' ')[0].strip()
    
    if not (url and pwd):
         # Fallback parsing strategy for different BaiduPCS output versions
         parts = output.split()
         for i, p in enumerate(parts):
             if 'http' in p: url = p
             if len(p) == 4 and i > 0 and parts[i-1] == '密码:': pwd = p

    # ------------------- FINAL CLEANUP -------------------
    if progress_callback: progress_callback("Cleaning up workspace...")
    try:
        if os.path.exists(LOCAL_INPUT_DIR): shutil.rmtree(LOCAL_INPUT_DIR)
        if os.path.exists(LOCAL_OUTPUT_DIR): shutil.rmtree(LOCAL_OUTPUT_DIR)
        if os.path.exists(LOCAL_ZIP_PATH): os.remove(LOCAL_ZIP_PATH)
    except Exception as e:
        print(f"Cleanup warning: {e}")

    return {
        "url": f"{url}?pwd={pwd}",
        "duration": math.ceil(total_seconds_accumulated),
        "errors": failed_files
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    try:
        result_data = run_job(args.job_id, args.source_url)
        print("✅ All done!")
        print(json.dumps(result_data))
    except Exception as e:
        print("❌ Failed:", e)
        raise
