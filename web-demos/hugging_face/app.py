"""
Minimal working WirePainter app that handles mask videos
This version includes progress bar, dual outputs, and high-quality MOV downloads
"""
import gradio as gr
import cv2
import numpy as np
import subprocess
import sys
import os
from pathlib import Path
import shutil
import warnings
import tempfile

# Suppress the MOV to MP4 conversion warnings
warnings.filterwarnings("ignore", message="Video does not have browser-compatible container or codec")

def create_mp4_preview(video_path):
    """Create MP4 preview for MOV files"""
    if not video_path:
        return None
        
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    
    # If it's already MP4, return as is
    if video_path.suffix.lower() == '.mp4':
        return str(video_path)
    
    # For MOV files, create MP4 preview
    if video_path.suffix.lower() in ['.mov', '.avi', '.mkv']:
        try:
            # Create preview in temp directory
            preview_dir = Path("temp_previews")
            preview_dir.mkdir(exist_ok=True)
            
            # Use simple filename to avoid encoding issues
            import hashlib
            video_hash = hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:8]
            preview_path = preview_dir / f"preview_{video_hash}.mp4"
            
            # Check if preview already exists
            if preview_path.exists():
                return str(preview_path)
            
            print(f"Creating MP4 preview for: {video_path}")
            
            # Quick conversion for preview with encoding handling
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-c:a', 'aac',
                '-movflags', '+faststart',
                '-y', str(preview_path)
            ]
            
            # Run with proper encoding handling
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace'  # Replace invalid characters instead of failing
            )
            
            if result.returncode == 0:
                print(f"Preview created: {preview_path}")
                # Store original path as metadata
                metadata_file = preview_path.with_suffix('.original')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    f.write(str(video_path))
                return str(preview_path)
            else:
                print(f"FFmpeg error: {result.stderr}")
                return str(video_path)
                
        except subprocess.SubprocessError as e:
            print(f"Subprocess error: {e}")
            # Try alternative method using cv2
            try:
                return create_preview_with_cv2(video_path, preview_dir)
            except:
                return str(video_path)
                
        except Exception as e:
            print(f"Error creating preview: {e}")
            return str(video_path)
    
    return str(video_path)

def create_preview_with_cv2(video_path, preview_dir):
    """Alternative preview creation using OpenCV"""
    print(f"Trying OpenCV method for preview creation")
    
    # Generate simple filename
    import hashlib
    video_hash = hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:8]
    preview_path = preview_dir / f"preview_cv2_{video_hash}.mp4"
    
    if preview_path.exists():
        return str(preview_path)
    
    # Read video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video with OpenCV")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
    
    # Copy frames
    frame_count = 0
    max_frames = 300  # Limit preview to first 300 frames for speed
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"Preview created with OpenCV: {preview_path}")
    
    # Store original path
    metadata_file = preview_path.with_suffix('.original')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(str(video_path))
    
    return str(preview_path)

def get_original_path(preview_path):
    """Get original video path from preview"""
    if not preview_path:
        return None
        
    preview_path = Path(preview_path)
    
    # Check if this is a preview file
    if "preview_" in preview_path.name:
        # Look for original path metadata
        original_file = preview_path.with_suffix('.original')
        if original_file.exists():
            try:
                with open(original_file, 'r', encoding='utf-8') as f:
                    original_path = Path(f.read().strip())
                    if original_path.exists():
                        return str(original_path)
            except:
                pass
    
    return str(preview_path)

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

def process_video(input_video_preview, mask_video_preview, mask_dilation, progress=gr.Progress()):
    """Process video with WirePainter"""
    
    if not input_video_preview or not mask_video_preview:
        return None, None, None, None
    
    try:
        progress(0.1, desc="Starting...")
        
        # Get original video paths
        input_video = get_original_path(input_video_preview)
        mask_video = get_original_path(mask_video_preview)
        
        print(f"Input preview: {input_video_preview}")
        print(f"Input original: {input_video}")
        print(f"Mask preview: {mask_video_preview}")
        print(f"Mask original: {mask_video}")
        
        # Log input file info
        input_path = Path(input_video)
        print(f"Processing input file: {input_path}")
        print(f"Input file extension: {input_path.suffix}")
        print(f"Input file size: {input_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Create persistent working directory
        work_dir = Path("propainter_workspace")
        work_dir.mkdir(exist_ok=True)
        
        # Clear previous runs
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(exist_ok=True)
        
        # Copy input video to workspace
        input_ext = input_path.suffix
        work_input = work_dir / f"input{input_ext}"
        shutil.copy(input_video, work_input)
        print(f"Copied original {input_ext} file to: {work_input}")
        
        progress(0.3, desc="Extracting mask frames...")
        
        # Extract mask frames from original mask video
        mask_dir = work_dir / "masks"
        print(f"Extracting mask frames to {mask_dir}")
        num_masks = extract_mask_frames(mask_video, mask_dir)
        print(f"Extracted {num_masks} mask frames")
        
        # Verify mask files exist
        mask_files = list(mask_dir.glob("*.png"))
        if not mask_files:
            return None, None, None, None
        
        progress(0.5, desc="Running WirePainter...")
        
        # Build WirePainter command
        cmd = [
            sys.executable,
            "../../inference_propainter.py",
            "--video", str(work_input),
            "--mask", str(mask_dir),
            "--output", str(work_dir),
            "--mask_dilation", str(mask_dilation),
            "--subvideo_length", "10",
            "--raft_iter", "50",
            "--ref_stride", "10",
            "--neighbor_length", "10"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run WirePainter with encoding handling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            error_msg = f"WirePainter failed:\n{result.stderr}\n{result.stdout}"
            print(error_msg)
            return None, None, None, None
        
        progress(0.9, desc="Finding outputs...")
        
        # Find output files
        output_dir_path = work_dir / "input"
        output_files = list(output_dir_path.glob("inpaint_out.*"))
        masked_files = list(output_dir_path.glob("masked_in.*"))
        
        if not output_files:
            print(f"No output files found in {output_dir_path}")
            return None, None, None, None
        
        output_file = output_files[0]
        masked_in_file = masked_files[0] if masked_files else None
        
        print(f"Found output_file: {output_file}")
        print(f"Found masked_in_file: {masked_in_file}")
        
        if output_file.exists():
            # Create output directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Copy to outputs directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get the output extension
            output_ext = output_file.suffix
            
            # Save outputs
            final_output = output_dir / f"{timestamp}_output{output_ext}"
            shutil.copy(output_file, final_output)
            print(f"final_output: {final_output}")
            
            final_masked_in = None
            if masked_in_file and masked_in_file.exists():
                final_masked_in = output_dir / f"{timestamp}_masked_in{output_ext}"
                shutil.copy(masked_in_file, final_masked_in)
                print(f"final_masked_in: {final_masked_in}")
            
            # Save original inputs
            input_ext = Path(input_video).suffix
            final_input = output_dir / f"{timestamp}_input{input_ext}"
            shutil.copy(input_video, final_input)
            
            mask_ext = Path(mask_video).suffix
            final_mask = output_dir / f"{timestamp}_mask{mask_ext}"
            shutil.copy(mask_video, final_mask)
            
            progress(1.0, desc="Complete!")
            
            # Create previews for output if needed
            preview_output = create_mp4_preview(str(final_output))
            preview_masked = create_mp4_preview(str(final_masked_in)) if final_masked_in else None
            
            # Return paths
            download_output = str(final_output) if output_ext == '.mov' else None
            download_masked = str(final_masked_in) if final_masked_in and output_ext == '.mov' else None
            
            return (preview_output, 
                    preview_masked, 
                    download_output, 
                    download_masked)
        else:
            return None, None, None, None
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None

# Create Gradio interface
with gr.Blocks(title="WirePainter") as demo:
    gr.Markdown("""
    # WirePainter - Stunt Wire Inpainting Demo
    
    Upload a video and a mask video. White pixels in the mask will be inpainted.
    
    **Note:** 
    - MOV/AVI files are automatically converted to MP4 for preview
    - Processing always uses your original high-quality files
    - ProRes MOV downloads available when MOV output is generated
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(
                label="Input Video", 
                sources=["upload"]
            )
            mask_video = gr.Video(
                label="Mask Video (white=inpaint, black=keep)", 
                sources=["upload"]
            )
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
            output_video = gr.Video(label="Inpainted Output (Preview)")
            masked_in_video = gr.Video(label="Masked Input (Preview)")
            
            # Simple download buttons for high-quality MOV files
            gr.Markdown("### High-Quality Downloads (when available):")
            download_output_btn = gr.File(label="Download Inpainted Output (ProRes MOV)", visible=True)
            download_masked_btn = gr.File(label="Download Masked Input (ProRes MOV)", visible=True)
    
    # Handle video uploads to create previews
    input_video.change(
        fn=create_mp4_preview,
        inputs=[input_video],
        outputs=[input_video]
    )
    
    mask_video.change(
        fn=create_mp4_preview,
        inputs=[mask_video],
        outputs=[mask_video]
    )
    
    process_btn.click(
        process_video,
        inputs=[input_video, mask_video, mask_dilation],
        outputs=[output_video, masked_in_video, download_output_btn, download_masked_btn]
    )

if __name__ == "__main__":
    # Clean up old previews on startup
    preview_dir = Path("temp_previews")
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
