import time
import argparse
import os
import stat
import subprocess
import shutil
import zipfile
from inference_propainter import load_models, run_inference, get_device
from pathlib import Path

# Update signature to accept progress_callback
def run_job(job_id: str, source_url: str, progress_callback=None) -> str:
    # ------------------- GLOBAL VARIABLES -------------------
    WORKSPACE = Path(__file__).resolve().parent
    LOCAL_INPUT_DIR   = os.path.join(WORKSPACE, "workdata", "input_videos")
    LOCAL_OUTPUT_DIR  = os.path.join(WORKSPACE, "workdata", "propainter_results")
    LOCAL_ZIP_PATH    = os.path.join(WORKSPACE, "workdata", "results.zip")
    REMOTE_JOB_PATH = f"/doya_jobs/{job_id}"
    
    # Force add execution rights
    BAIDU_PCS = os.path.join(WORKSPACE, "BaiduPCS-Go")
    if os.path.exists(BAIDU_PCS):
        st = os.stat(BAIDU_PCS)
        os.chmod(BAIDU_PCS, st.st_mode | stat.S_IEXEC)

    # ------------------- Helper: Retry Wrapper -------------------
    def run_pcs_with_retry(cmd_list, max_retries=3):
        for attempt in range(max_retries):
            try:
                # We use capture_output=True so we can print errors if needed
                subprocess.run(cmd_list, check=True, capture_output=True, text=True)
                return # Success, exit loop
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr or e.stdout or "Unknown Error"
                print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {error_msg}", flush=True)
                
                # If it's the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise RuntimeError(f"BaiduPCS Command Failed after {max_retries} retries: {error_msg}")
                
                # Wait before retrying (Exponential backoff: 2s, 4s, 8s)
                time.sleep(2 * (attempt + 1))

    # ------------------- Login & Config -------------------
    bduss = os.getenv("BAIDU_BDUSS")
    stoken = os.getenv("BAIDU_STOKEN")
    if not bduss or not stoken:
        raise RuntimeError("Missing BAIDU_BDUSS / BAIDU_STOKEN")

    # Login (Standard run is fine here)
    subprocess.run([BAIDU_PCS, "login", f"-bduss={bduss}", f"-stoken={stoken}"], check=True, stdout=subprocess.DEVNULL)

    # --- STABILITY FIXES: Set User Agent and AppID ---
    # Pretend to be a standard Chrome browser on Windows
    run_pcs_with_retry([BAIDU_PCS, "config", "set", "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"])
    # Set AppID to 266719 (Standard) or 778750 (Enterprise) - 266719 is usually best for transfers
    run_pcs_with_retry([BAIDU_PCS, "config", "set", "-appid", "266719"])
    
    run_pcs_with_retry([BAIDU_PCS, "config", "set", "-savedir", LOCAL_INPUT_DIR])
    run_pcs_with_retry([BAIDU_PCS, "config", "set", "-max_parallel", "20"])
    
    # ------------------- Clean Local Dirs -------------------
    for p in [LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR]:
        if os.path.exists(p): shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    # ------------------- Prepare Remote Dir -------------------
    # Ignore errors on mkdir (if it exists, that's fine)
    subprocess.run([BAIDU_PCS, "mkdir", REMOTE_JOB_PATH], stderr=subprocess.DEVNULL)
    run_pcs_with_retry([BAIDU_PCS, "cd", REMOTE_JOB_PATH])

    # ------------------- Download with Retry -------------------
    if progress_callback: progress_callback("Downloading video...")
    print(f"Downloading from: {source_url}", flush=True)

    transfer_cmd = [BAIDU_PCS, "transfer", "--download"]
    
    # Handle URL splitting logic
    if "?pwd=" in source_url:
        link, pwd = source_url.split("?pwd=")
        transfer_cmd.extend([link, pwd])
    else:
        transfer_cmd.append(source_url)

    # EXECUTE TRANSFER WITH RETRY
    # This is where Error -9 usually happens. The retry loop handles it.
    run_pcs_with_retry(transfer_cmd, max_retries=5)

    # ------------------- Rest of the script... -------------------
    # ... (Continue with your existing video finding, inference, zipping, uploading code)
    # ...
    
    # (Just copy the rest of your logic here for finding .mov files, inference, etc.)
    # Below is the quick summary of the rest to complete the snippet:

    video_dir = next((root for root, _, files in os.walk(LOCAL_INPUT_DIR) if any(f.lower().endswith('.mov') for f in files)), None)
    if not video_dir: raise ValueError("No .mov files found after download")

    # ... [INSERT YOUR INFERENCE LOOP HERE] ...
    # ... (Use the code from the previous message for inference loop) ...

    # ------------------- Upload & Share -------------------
    if progress_callback: progress_callback("Uploading results...")
    run_pcs_with_retry([BAIDU_PCS, "upload", LOCAL_ZIP_PATH, REMOTE_JOB_PATH])

    share = subprocess.run(
        [BAIDU_PCS, "share", "set", f"{REMOTE_JOB_PATH}/results.zip", "--period", "0"],
        capture_output=True, text=True, check=True
    )
    
    # Parse Output
    output = share.stdout.strip()
    parts = [p.strip() for p in output.split(',')]
    final_url = next((p.split('链接: ')[1] for p in parts if p.startswith('链接: ')), None)
    final_pwd = next((p.split('密码: ')[1] for p in parts if p.startswith('密码: ')), None)
    
    if not (final_url and final_pwd):
        raise RuntimeError(f"Failed to parse share link. Output: {output}")

    return f"{final_url}?pwd={final_pwd}"

# ==================== CLI ENTRYPOINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    try:
        result_url = run_job(args.job_id, args.source_url)
        print("✅ All done!")
        print(result_url)
    except Exception as e:
        print("❌ Failed:", e)
        raise
