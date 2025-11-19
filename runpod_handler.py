import sys
import traceback
from runpod import serverless
from batch_inference import run_job

def handler(job):
    inp = job["input"]
    job_id = inp.get("jobID")
    source_url = inp.get("sourceUrl")

    if not job_id or not source_url:
        return {"status": 0, "error": "Missing jobID or sourceUrl"}

    try:
        result_url = run_job(job_id=job_id, source_url=source_url)

        return {
            "status": 1,
            "jobID": job_id,
            "resultUrl": result_url
        }
    except Exception as e:
        print(f"❌ Job {job_id} Failed!", flush=True)
        traceback.print_exc() # This prints the full stack trace to logs
        return {
            "status": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    serverless.start({"handler": handler})
