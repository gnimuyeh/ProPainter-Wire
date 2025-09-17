# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import tempfile
import subprocess
import json
import shutil
import torch
import torchvision
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def probe_video_info(video_path):
    """Probe video and return metadata"""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=pix_fmt,color_range,color_space,color_transfer,color_primaries,r_frame_rate',
        '-of', 'json', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        ref_info = json.loads(result.stdout)['streams'][0]
        
        r_frame_rate = ref_info.get('r_frame_rate', '24/1')
        try:
            fps = eval(r_frame_rate)
        except Exception:
            fps = 24.0
            
        return {
            'fps': fps,
            'pix_fmt': ref_info.get('pix_fmt', 'yuv444p10le'),
            'color_range': ref_info.get('color_range', 'tv'),
            'color_space': ref_info.get('color_space', 'bt709'),
            'color_transfer': ref_info.get('color_transfer', 'bt709'),
            'color_primaries': ref_info.get('color_primaries', 'bt709')
        }
    return {'fps': 24.0}

def extract_frames_to_png(video_path, output_dir, is_mask=False):
    """Extract frames to 16-bit PNG with maximum compression"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if is_mask:
        pix_fmt = 'gray16le'
        pattern = 'mask_%06d.png'
    else:
        pix_fmt = 'rgb48le'  # Always 16-bit for preserving 12-bit data
        pattern = 'frame_%06d.png'
    
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'image2',
        '-c:v', 'png',
        '-pix_fmt', pix_fmt,
        '-compression_level', '9',  # Always max compression
        str(output_dir / pattern),
        '-y', '-loglevel', 'error'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")
    
    return len(list(output_dir.glob('*.png')))

def load_frames_preserve_originals(frame_dir):
    """Load frames preserving originals and creating processing versions"""
    frames_original = []
    frames_processing = []
    
    frame_files = sorted(Path(frame_dir).glob('*.png'))
    
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        
        frames_original.append(frame)
        
        # Convert to 8-bit for processing
        if frame.dtype == np.uint16:
            frame_8bit = (frame / 256).astype(np.uint8)
        else:
            frame_8bit = frame
        
        # BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_BGR2RGB)
        frames_processing.append(Image.fromarray(frame_rgb))
    
    return frames_original, frames_processing

def resize_frames(frames, size=None):
    """Resize frames to be divisible by 8 for model processing"""
    if size is not None:
        out_size = size
    else:
        out_size = frames[0].size
    
    process_size = (out_size[0] - out_size[0]%8, out_size[1] - out_size[1]%8)
    
    if out_size != process_size:
        frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size

def load_masks(mask_path, num_frames, size, dilation=0):
    """Load and process masks with optional dilation"""
    masks = []
    
    # Load mask images
    if os.path.isdir(mask_path):
        mask_files = sorted(Path(mask_path).glob('*.png'))
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if mask.dtype == np.uint16:
                mask = (mask / 256).astype(np.uint8)
            masks.append(Image.fromarray(mask))
    elif mask_path.endswith(('.png', '.jpg', '.jpeg')):
        masks = [Image.open(mask_path)]
    else:
        # Video mask - extract frames
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_frames_to_png(mask_path, temp_dir, is_mask=True)
            for mask_file in sorted(Path(temp_dir).glob('*.png')):
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask.dtype == np.uint16:
                    mask = (mask / 256).astype(np.uint8)
                masks.append(Image.fromarray(mask))
    
    # Process masks
    processed_masks = []
    for mask_img in masks:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_array = np.array(mask_img.convert('L'))
        
        # Binarize
        binary_mask = (mask_array > 127).astype(np.uint8)
        
        # Apply dilation if requested
        if dilation > 0:
            binary_mask = scipy.ndimage.binary_dilation(binary_mask, iterations=dilation).astype(np.uint8)
        
        processed_masks.append(Image.fromarray(binary_mask * 255))
    
    # Repeat single mask for all frames if needed
    if len(processed_masks) == 1 and num_frames > 1:
        processed_masks = processed_masks * num_frames
    
    return processed_masks, processed_masks  # Return same masks for flow and regular

def save_video_prores(frames_original, comp_frames, masks, output_path, fps, metadata=None):
    """Save as ProRes 4444 MOV (10-bit limitation due to FFmpeg)"""
    print(f"\n🎬 Saving ProRes 4444 video...")
    print(f"  ⚠️  Note: FFmpeg only supports 10-bit ProRes, not 12-bit")
    print(f"  For true 12-bit, use PNG sequence → DaVinci Resolve")
    
    h_orig, w_orig = frames_original[0].shape[:2]
    num_frames = len(comp_frames)
    
    # Prepare FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{w_orig}x{h_orig}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'prores_ks',
        '-profile:v', '4',
        '-pix_fmt', 'yuv444p10le',  # 10-bit (FFmpeg limitation)
        '-vendor', 'apl0'
    ]
    
    # Add color metadata if available
    if metadata:
        cmd.extend([
            '-color_primaries', metadata.get('color_primaries', 'bt709'),
            '-color_trc', metadata.get('color_transfer', 'bt709'),
            '-colorspace', metadata.get('color_space', 'bt709')
        ])
    
    cmd.extend(['-movflags', '+write_colr+faststart', output_path])
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        for i in tqdm(range(num_frames), desc="Writing frames"):
            # Get mask
            mask_tensor = masks[0, i, 0].cpu().numpy()
            mask_bool = mask_tensor > 0.5
            
            # Resize mask if needed
            if mask_bool.shape != (h_orig, w_orig):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w_orig, h_orig),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Get original and inpainted frames
            original = frames_original[i]
            inpainted = comp_frames[i]
            
            # Resize inpainted if needed
            if inpainted.shape[:2] != (h_orig, w_orig):
                inpainted = cv2.resize(inpainted, (w_orig, h_orig))
            
            # Convert inpainted RGB to BGR
            inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
            
            # Composite at highest available precision
            if original.dtype == np.uint16:
                # Convert to 8-bit for FFmpeg input
                original_8bit = (original / 256).astype(np.uint8)
                result = original_8bit.copy()
                for c in range(3):
                    result[:,:,c][mask_bool] = inpainted_bgr[:,:,c][mask_bool]
            else:
                result = original.copy()
                for c in range(3):
                    result[:,:,c][mask_bool] = inpainted_bgr[:,:,c][mask_bool]
            
            # Write frame
            proc.stdin.write(result.tobytes())
        
        proc.stdin.close()
        returncode = proc.wait()
        
        if returncode != 0:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
            
    finally:
        proc.stderr.close()
        if proc.poll() is None:
            proc.terminate()
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"  Format: ProRes 4444 (10-bit)")

def save_png_sequence(frames_original, comp_frames, masks, output_dir, 
                      save_metadata=False, metadata=None):
    """Save pixel-perfect PNG sequence with maximum compression"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    num_frames = len(comp_frames)
    h_orig, w_orig = frames_original[0].shape[:2]
    
    print(f"\n📸 Saving PNG sequence ({num_frames} frames)...")
    print(f"  Resolution: {w_orig}x{h_orig} (16-bit for 12-bit preservation)")
    print(f"  Compression: Maximum (level 9, lossless)")
    
    for i in tqdm(range(num_frames), desc="Writing frames"):
        # Get mask
        mask_tensor = masks[0, i, 0].cpu().numpy()
        mask_bool = mask_tensor > 0.5
        
        # Resize mask if needed
        if mask_bool.shape != (h_orig, w_orig):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w_orig, h_orig),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Get original and inpainted frames
        original = frames_original[i]
        inpainted = comp_frames[i]
        
        # Resize inpainted if needed
        if inpainted.shape[:2] != (h_orig, w_orig):
            inpainted = cv2.resize(inpainted, (w_orig, h_orig))
        
        # Convert inpainted RGB to BGR
        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
        
        # Match bit depths
        if original.dtype == np.uint16:
            inpainted_16bit = inpainted_bgr.astype(np.uint16) * 256
            result = original.copy()
            for c in range(3):
                result[:,:,c][mask_bool] = inpainted_16bit[:,:,c][mask_bool]
        else:
            result = original.copy()
            for c in range(3):
                result[:,:,c][mask_bool] = inpainted_bgr[:,:,c][mask_bool]
        
        # Save with maximum compression
        cv2.imwrite(
            str(output_dir / f"frame_{i:06d}.png"),
            result,
            [cv2.IMWRITE_PNG_COMPRESSION, 9]  # Always max compression
        )
    
    # Save metadata if requested
    if save_metadata and metadata:
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved")
    
    print(f"\n✅ Saved to: {output_dir}")
    return output_dir

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """Get reference frame indices for ProPainter"""
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids and len(ref_index) < ref_num:
                ref_index.append(i)
    return ref_index

def load_models(device, use_half=False):
    """Load ProPainter models"""
    # RAFT
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'raft-things.pth'),
        model_dir='weights', progress=True, file_name=None
    )
    fix_raft = RAFT_bi(ckpt_path, device)
    
    # Flow completion
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
        model_dir='weights', progress=True, file_name=None
    )
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    fix_flow_complete.requires_grad_(False)
    fix_flow_complete.to(device)
    fix_flow_complete.eval()
    
    # Inpainting model
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
        model_dir='weights', progress=True, file_name=None
    )
    model = InpaintGenerator(model_path=ckpt_path)
    model.to(device)
    model.eval()
    
    if use_half:
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
    
    return fix_raft, fix_flow_complete, model

def compute_optical_flow(frames_tensor, fix_raft, raft_iter):
    """Compute optical flow for video frames"""
    video_length = frames_tensor.size(1)
    w = frames_tensor.shape[-1]
    
    # Determine clip length based on resolution
    if w <= 640:
        short_clip_len = 12
    elif w <= 720:
        short_clip_len = 8
    elif w <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2
    
    if video_length <= short_clip_len:
        return fix_raft(frames_tensor, iters=raft_iter)
    
    # Process in chunks for longer videos
    flows_f, flows_b = [], []
    for f in range(0, video_length, short_clip_len):
        end_f = min(video_length, f + short_clip_len)
        start_f = max(0, f - 1) if f > 0 else f
        
        flows = fix_raft(frames_tensor[:, start_f:end_f], iters=raft_iter)
        flows_f.append(flows[0])
        flows_b.append(flows[1])
        torch.cuda.empty_cache()
    
    return torch.cat(flows_f, dim=1), torch.cat(flows_b, dim=1)

def run_propainter_inference(frames, masks, models, device, args):
    """Run ProPainter inference pipeline"""
    fix_raft, fix_flow_complete, model = models
    use_half = args.fp16 and device != torch.device('cpu')
    
    frames_tensor = to_tensors()(frames).unsqueeze(0) * 2 - 1
    masks_tensor = to_tensors()(masks).unsqueeze(0)
    
    frames_tensor = frames_tensor.to(device)
    masks_tensor = masks_tensor.to(device)
    
    video_length = frames_tensor.size(1)
    h, w = frames_tensor.shape[-2:]
    
    print(f'\n🎨 Processing {video_length} frames...')
    
    with torch.no_grad():
        # Compute optical flow
        gt_flows = compute_optical_flow(frames_tensor, fix_raft, args.raft_iter)
        torch.cuda.empty_cache()
        
        if use_half:
            frames_tensor = frames_tensor.half()
            masks_tensor = masks_tensor.half()
            gt_flows = (gt_flows[0].half(), gt_flows[1].half())
        
        # Complete flow
        flow_length = gt_flows[0].size(1)
        if flow_length > args.subvideo_length:
            # Handle long videos in chunks
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                
                flows_sub = (gt_flows[0][:, s_f:e_f], gt_flows[1][:, s_f:e_f])
                masks_sub = masks_tensor[:, s_f:e_f+1]
                
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(flows_sub, masks_sub)
                pred_flows_bi_sub = fix_flow_complete.combine_flow(flows_sub, pred_flows_bi_sub, masks_sub)
                
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
            
            pred_flows = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
        else:
            pred_flows, _ = fix_flow_complete.forward_bidirect_flow(gt_flows, masks_tensor)
            pred_flows = fix_flow_complete.combine_flow(gt_flows, pred_flows, masks_tensor)
        torch.cuda.empty_cache()
        
        # Image propagation
        masked_frames = frames_tensor * (1 - masks_tensor)
        b, t, _, _, _ = masks_tensor.size()
        prop_imgs, updated_masks = model.img_propagation(masked_frames, pred_flows, masks_tensor, 'nearest')
        updated_frames = frames_tensor * (1 - masks_tensor) + prop_imgs.view(b, t, 3, h, w) * masks_tensor
        updated_masks = updated_masks.view(b, t, 1, h, w)
        torch.cuda.empty_cache()
    
    # Feature propagation + transformer
    comp_frames = [None] * video_length
    neighbor_stride = args.neighbor_length // 2
    ref_num = args.subvideo_length // args.ref_stride if video_length > args.subvideo_length else -1
    
    for f in tqdm(range(0, video_length, neighbor_stride), desc="Inpainting"):
        neighbor_ids = list(range(
            max(0, f - neighbor_stride),
            min(video_length, f + neighbor_stride + 1)
        ))
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        
        with torch.no_grad():
            # Select frames and masks
            selected_ids = neighbor_ids + ref_ids
            selected_frames = updated_frames[:, selected_ids]
            selected_masks = masks_tensor[:, selected_ids]
            selected_update_masks = updated_masks[:, selected_ids]
            selected_flows = (pred_flows[0][:, neighbor_ids[:-1]], 
                            pred_flows[1][:, neighbor_ids[:-1]])
            
            # Run model
            l_t = len(neighbor_ids)
            pred_img = model(selected_frames, selected_flows, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            
            # Merge results
            for i, idx in enumerate(neighbor_ids):
                img = pred_img[i].astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = ((comp_frames[idx].astype(np.uint16) + 
                                       img.astype(np.uint16)) // 2).astype(np.uint8)
        
        torch.cuda.empty_cache()
    
    return comp_frames, masks_tensor

def run_inference(video, mask, output, models=None, **kwargs):
    """Main inference function"""
    # Set defaults for any missing arguments
    defaults = {
        'resize_ratio': 1.0,
        'height': -1,
        'width': -1,
        'mask_dilation': 4,
        'ref_stride': 10,
        'neighbor_length': 10,
        'subvideo_length': 80,
        'raft_iter': 20,
        'fp16': False,
        'save_metadata': False,
        'output_video': False  # New flag for video output
    }
    
    # Merge defaults with provided kwargs
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    
    args = argparse.Namespace(**kwargs)
    device = get_device()
    
    # Setup paths
    input_is_video = not os.path.isdir(video)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Extract or load frames
        if input_is_video:
            video_info = probe_video_info(video)
            fps = video_info['fps']
            
            input_frames_dir = temp_dir / 'input_frames'
            print(f"\n📹 Extracting frames from video...")
            extract_frames_to_png(video, input_frames_dir)
            
            metadata = video_info if args.save_metadata else None
        else:
            input_frames_dir = Path(video)
            fps = 24.0
            metadata = None
        
        # Load frames
        print(f"📁 Loading frames...")
        frames_original, frames = load_frames_preserve_originals(input_frames_dir)
        
        # Handle resizing
        size = frames[0].size
        if args.width > 0 and args.height > 0:
            size = (args.width, args.height)
        elif args.resize_ratio != 1.0:
            size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
        
        frames, process_size, out_size = resize_frames(frames, size)
        
        # Load masks
        flow_masks, masks = load_masks(mask, len(frames), process_size, args.mask_dilation)
        
        # Run ProPainter
        comp_frames, masks_tensor = run_propainter_inference(
            frames, masks, models, device, args
        )
        
        # Resize output frames to target size
        comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        
        # Save output
        if args.output_video:
            # Save as ProRes MOV (10-bit limitation)
            save_video_prores(
                frames_original, comp_frames, masks_tensor,
                output, fps, metadata=metadata
            )
            print("\n⚠️  Note: ProRes output is limited to 10-bit by FFmpeg")
            print("  For 12-bit preservation, use PNG sequence output instead")
        else:
            # Save as PNG sequence (preserves 12-bit)
            save_png_sequence(
                frames_original, comp_frames, masks_tensor,
                output, save_metadata=args.save_metadata, metadata=metadata
            )
            print("\nNext: Import PNGs into DaVinci Resolve/Nuke for 12-bit ProRes 4444 export")
        
        print("\n🎉 Processing complete!")
        print(f"Output: {output}")
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input/Output
    parser.add_argument('-i', '--video', type=str, required=True,
                       help='Input video or image folder')
    parser.add_argument('-m', '--mask', type=str, required=True,
                       help='Mask video, image, or folder')
    parser.add_argument('-o', '--output', type=str, default='output_frames',
                       help='Output PNG sequence directory or video path')
    
    # Processing options
    parser.add_argument('--resize_ratio', type=float, default=1.0)
    parser.add_argument('--height', type=int, default=-1)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--mask_dilation', type=int, default=4,
                       help='Mask dilation iterations')
    
    # ProPainter parameters
    parser.add_argument('--ref_stride', type=int, default=10)
    parser.add_argument('--neighbor_length', type=int, default=10)
    parser.add_argument('--subvideo_length', type=int, default=80)
    parser.add_argument('--raft_iter', type=int, default=20)
    parser.add_argument('--fp16', action='store_true',
                       help='Use half precision')
    
    # Output options
    parser.add_argument('--output_video', action='store_false',
                       help='Output ProRes MOV video instead of PNG sequence (10-bit limitation)')
    parser.add_argument('--save_metadata', action='store_false',
                       help='Save metadata.json with color space info')
    
    args = parser.parse_args()
    
    # Load models
    device = get_device()
    models = load_models(device, args.fp16)
    
    # Run inference
    run_inference(**vars(args), models=models)
