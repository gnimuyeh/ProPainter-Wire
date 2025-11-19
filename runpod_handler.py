import sys
import traceback
import runpod
from batch_inference import run_job

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

def handler(job):
    # 1. Use RunPod's native Request ID as the single truth
    runpod_id = job["id"]
    
    inp = job["input"]
    source_url = inp.get("sourceUrl")

    print(f"Received Job ID: {runpod_id}", flush=True)

    if not source_url:
        return {"status": 0, "error": "Missing sourceUrl"}

    # 2. Define the Progress Callback
    def update_progress(message):
        try:
            # This sends the message back to the UI while running
            runpod.serverless.progress_update(job, message)
        except Exception:
            pass

    try:
        # 3. Execute Logic
        result_url = run_job(
            job_id=runpod_id, 
            source_url=source_url, 
            progress_callback=update_progress
        )

        return {
            "status": 1,
            "jobID": runpod_id,
            "resultUrl": result_url
        }
    except Exception as e:
        print(f"❌ Job {runpod_id} Failed!", flush=True)
        traceback.print_exc()
        return {
            "status": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
