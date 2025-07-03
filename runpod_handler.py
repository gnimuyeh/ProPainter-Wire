from runpod import serverless
import subprocess
import os
import sys

def handler(job):
    """
    RunPod worker handler that receives job input,
    runs the ProPainter inference, and returns output path or result.

    Args:
      job (dict): RunPod job object with 'input' dict containing needed inputs.

    Returns:
      dict: Output dict to return to caller.
    """
    try:
        # Example: input expects paths or data from job["input"]
        input_video = job["input"].get("input_video")
        mask_video = job["input"].get("mask_video")
        mask_dilation = job["input"].get("mask_dilation", 4)
        ref_stride = job["input"].get("ref_stride", 10)
        neighbor_length = job["input"].get("neighbor_length", 10)
        subvideo_length = job["input"].get("subvideo_length", 80)
        raft_iter = job["input"].get("raft_iter", 20)


        # Paths inside container workspace, adjust as needed
        # Assume inputs are already uploaded to container or mounted volume

        # Build command to run your inference script
        cmd = [
            "python",
            "inference_propainter.py",
            "--video", input_video,
            "--mask", mask_video,
            "--mask_dilation", str(mask_dilation),
            "--ref_stride", str(ref_stride),
            "--neighbor_length", str(neighbor_length),
            "--subvideo_length", str(subvideo_length),
            "--raft_iter", str(raft_iter),
            "--output", "outputs",
            "--fp16"
        ]

        # Run the inference subprocess
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

        if result.returncode != 0:
            return {"error": result.stderr}

        # On success, return path or success message
        return {"output": "outputs/inpainted_output.mov"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    serverless.start({"handler": handler})
