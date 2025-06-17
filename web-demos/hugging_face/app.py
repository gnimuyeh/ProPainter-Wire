"""
Minimal working WirePainter app that handles mask videos
This version includes progress bar and no width/height settings
"""

import gradio as gr
import cv2
import numpy as np
import subprocess
import sys
import os
from pathlib import Path
import shutil

def extract_mask_frames(mask_video_path, output_dir):
    """Extract mask frames from video"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Use OpenCV to extract frames
    cap = cv2.VideoCapture(str(mask_video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open mask video: {mask_video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Binarize: >127 becomes 255 (white=inpaint), <=127 becomes 0 (black=keep)
        frame = (frame > 127).astype(np.uint8) * 255

        # Save frame
        frame_path = Path(output_dir) / f"{frame_idx:05d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

    cap.release()
    return frame_idx

def process_video(input_video, mask_video, mask_dilation, progress=gr.Progress()):
    """Process video with WirePainter"""

    if not input_video or not mask_video:
        return None

    try:
        progress(0.1, desc="Starting...")

        # Create persistent working directory
        work_dir = Path("propainter_workspace")
        work_dir.mkdir(exist_ok=True)

        # Clear previous runs
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(exist_ok=True)

        # Copy input video to workspace
        input_ext = Path(input_video).suffix
        work_input = work_dir / f"input{input_ext}"
        shutil.copy(input_video, work_input)

        progress(0.3, desc="Extracting mask frames...")

        # Extract mask frames
        mask_dir = work_dir / "masks"
        print(f"Extracting mask frames to {mask_dir}")
        num_masks = extract_mask_frames(mask_video, mask_dir)
        print(f"Extracted {num_masks} mask frames")

        # Verify mask files exist
        mask_files = list(mask_dir.glob("*.png"))
        if not mask_files:
            return None

        progress(0.5, desc="Running WirePainter...")

        # Build WirePainter command
        cmd = [
            sys.executable,
            "../../inference_propainter.py",
            "--video", str(work_input),
            "--mask", str(mask_dir),
            "--output", str(work_dir),
            "--mask_dilation", str(mask_dilation),
            "--subvideo_length 10",
            "--raft_iter 50",
            "--ref_stride 10",
            "--neighbor_length 10"
        ]

        print(f"Running: {' '.join(cmd)}")

        # Run WirePainter
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))  # Run from script directory
        )

        print(f"result: {result}")

        if result.returncode != 0:
            error_msg = f"WirePainter failed:\n{result.stderr}\n{result.stdout}"
            print(error_msg)
            return None

        progress(0.9, desc="Finding output...")

        # Output file is always named inpaint_out.mp4
        output_file = work_dir / "input/inpaint_out.mp4"

        print(f"output_file: {output_file}")

        if output_file.exists():
            # Create output directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # Copy to outputs directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output = output_dir / f"{timestamp}_output.mp4"
            shutil.copy(output_file, final_output)

            print(f"final_output: {final_output}")

            # Save input video with same timestamp
            input_ext = Path(input_video).suffix
            final_input = output_dir / f"{timestamp}_input{input_ext}"
            shutil.copy(input_video, final_input)
            print(f"Saved input video: {final_input}")

            # Save mask video with same timestamp
            mask_ext = Path(mask_video).suffix
            final_mask = output_dir / f"{timestamp}_mask{mask_ext}"
            shutil.copy(mask_video, final_mask)
            print(f"Saved mask video: {final_mask}")

            progress(1.0, desc="Complete!")
            return str(final_output)
        else:
            print(f"output_file not exist")
            return None

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

# Create Gradio interface
with gr.Blocks(title="WirePainter") as demo:
    gr.Markdown("""
    # WirePainter - Stunt Wire Inpainting Demo

    Upload a video and a mask video. White pixels in the mask will be inpainted.
    """)

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", sources=["upload"])
            mask_video = gr.Video(label="Mask Video (white=inpaint, black=keep)", sources=["upload"])
            mask_dilation = gr.Slider(
                minimum=0,
                maximum=20,
                value=0,
                step=1,
                label="Mask Dilation",
                info="Expand mask regions by this many pixels"
            )
            process_btn = gr.Button("Process", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Output")

    process_btn.click(
        process_video,
        inputs=[input_video, mask_video, mask_dilation],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
