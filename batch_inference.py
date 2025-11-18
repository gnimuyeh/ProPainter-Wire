import argparse
import os
import subprocess
import shutil
import zipfile
from inference_propainter import load_models, run_inference, get_device
from pathlib import Path

def run_job(job_id: str, source_url: str) -> str:
    # ------------------- GLOBAL VARIABLES -------------------
    # This makes everything work the same locally AND in Docker/RunPod
    WORKSPACE = Path(__file__).resolve().parent
    BAIDU_PCS         = os.path.join(WORKSPACE, "BaiduPCS-Go")        # ← binary name
    LOCAL_INPUT_DIR   = os.path.join(WORKSPACE, "workdata", "input_videos")
    LOCAL_OUTPUT_DIR  = os.path.join(WORKSPACE, "workdata", "propainter_results")
    LOCAL_ZIP_PATH    = os.path.join(WORKSPACE, "workdata", "results.zip")
    REMOTE_JOB_PATH = f"/doya_jobs/{job_id}"
    REMOTE_ZIP_PATH = f"{REMOTE_JOB_PATH}/results.zip"

    # ------------------- Ensure Baidu login (idempotent & secure) -------------------
    bduss = os.getenv("BAIDU_BDUSS")
    stoken = os.getenv("BAIDU_STOKEN")
    if not bduss or not stoken:
        raise RuntimeError("Missing BAIDU_BDUSS / BAIDU_STOKEN environment variables or secrets")
    subprocess.run([BAIDU_PCS, "login", f"-bduss={bduss}", f"-stoken={stoken}"], check=True)

    # ------------------- Clean & prepare local dirs -------------------
    for p in [LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    # ------------------- BaiduPCS-Go config & download -------------------
    subprocess.run([BAIDU_PCS, "config", "set", "-savedir", LOCAL_INPUT_DIR], check=True)
    subprocess.run([BAIDU_PCS, "config", "set", "-max_parallel", "20"], check=True)
    subprocess.run([BAIDU_PCS, "mkdir", REMOTE_JOB_PATH], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([BAIDU_PCS, "cd", REMOTE_JOB_PATH], check=True)
    subprocess.run([BAIDU_PCS, "transfer", "--download", source_url], check=True)

    # ------------------- Find video directory -------------------
    video_dir = next(
        (root for root, _, files in os.walk(LOCAL_INPUT_DIR)
         if any(f.lower().endswith('.mov') for f in files)),
        None
    )
    if not video_dir:
        raise ValueError("No .mov files found after download")

    # ------------------- Load models -------------------
    device = get_device()
    models = load_models(device, use_half=False)

    # ------------------- Process each video -------------------
    for file in os.listdir(video_dir):
        if not file.lower().endswith('.mov') or '_mask.' in file or '_result.' in file:
            continue

        basename, ext = os.path.splitext(file)
        mask_path = os.path.join(video_dir, f"{basename}_mask{ext}")
        if not os.path.exists(mask_path):
            print(f"Skipping {file} — no mask")
            continue

        final_out = os.path.join(LOCAL_OUTPUT_DIR, f"{basename}_result{ext}")
        if os.path.exists(final_out):
            continue

        binary_mask = os.path.join(video_dir, f"{basename}_mask_binary{ext}")

        # Binary mask conversion
        subprocess.run([
            "ffmpeg", "-y", "-i", mask_path,
            "-vf", "format=gray,geq='if(gt(p(X,Y),128),255,0)'",
            "-pix_fmt", "gray", "-c:v", "ffv1", binary_mask
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Inference
        run_inference(
            video=os.path.join(video_dir, file),
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

    # ------------------- Zip results -------------------
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(LOCAL_OUTPUT_DIR):
            for f in files:
                fp = os.path.join(root, f)
                arcname = os.path.relpath(fp, os.path.dirname(LOCAL_OUTPUT_DIR))
                z.write(fp, arcname)

    # ------------------- Upload & share -------------------
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

    return f"{url}?pwd={pwd}"

# ==================== CLI ENTRYPOINT (local testing) ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProPainter Baidu Pan Batch Job")
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
