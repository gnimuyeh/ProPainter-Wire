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
    """Probe video and return (color_range, color_space, color_transfer, color_primaries, fps, prores_profile_num, pix_fmt)."""
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_tag_string,pix_fmt,color_range,color_space,color_transfer,color_primaries,r_frame_rate,profile',
        '-of', 'json', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        ref_info = json.loads(result.stdout)['streams'][0]

        color_range = ref_info.get('color_range', 'tv')
        color_space = ref_info.get('color_space', 'bt709')
        color_transfer = ref_info.get('color_transfer', 'bt709')
        color_primaries = ref_info.get('color_primaries', 'bt709')
        r_frame_rate = ref_info.get('r_frame_rate', '24/1')
        try:
            fps = eval(r_frame_rate)
        except Exception:
            fps = 24.0

        codec_tag = (ref_info.get('codec_tag_string') or '').lower()
        pix_fmt = ref_info.get('pix_fmt', 'yuv444p10le')
        profile_field = (ref_info.get('profile') or '').lower()

        codec_tag_to_profile = {
            'apco': '0', 'apcs': '1', 'apcn': '2', 'apch': '3', 'ap4h': '4', 'ap4x': '5',
        }

        profile = codec_tag_to_profile.get(codec_tag, '4')

        transfer_l = (color_transfer or '').lower()
        if transfer_l.startswith('gamma'):
            color_transfer = 'bt709'

        print(f"Detected codec_tag={codec_tag} -> profile={profile}, pix_fmt={pix_fmt}, color_trc={color_transfer}")
        return color_range, color_space, color_transfer, color_primaries, fps, profile, pix_fmt

    return 'tv', 'bt709', 'bt709', 'bt709', 24.0, '4', 'yuv444p10le'

def save_video_highest_quality(frames, output_path, fps, reference_video=None, original_frames=None, masks=None):
    if not frames:
        raise ValueError("No frames to save")

    h, w, _ = frames[0].shape
    n_frames = len(frames)

    if reference_video and os.path.isfile(reference_video):
        color_range, color_space, color_transfer, color_primaries, _, profile, _ = probe_video_info(reference_video)
    else:
        color_range, color_space, color_transfer, color_primaries = 'tv','bt709','bt709','bt709'

    profile = '4'
    pix_fmt = 'yuv444p10le'

    if original_frames is not None and masks is not None:
        if not (len(original_frames) == len(masks) == n_frames):
            raise ValueError(f"Length mismatch: frames={n_frames}, original_frames={len(original_frames)}, masks={len(masks)}")
        blended_frames = []
        for f, orig, mask in zip(frames, original_frames, masks):
            mask_bool = mask.astype(bool)
            frame_copy = orig.copy()
            frame_copy[mask_bool] = f[mask_bool]
            blended_frames.append(frame_copy)
        frames_to_write = blended_frames
    else:
        frames_to_write = frames

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
        '-c:v', 'prores_ks', '-profile:v', profile, '-pix_fmt', pix_fmt, '-vendor', 'apl0',
        '-color_primaries', color_primaries, '-color_trc', color_transfer, '-colorspace', color_space,
        '-movflags', '+write_colr+faststart', output_path
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in tqdm(frames_to_write):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        returncode = proc.wait()
        if returncode != 0:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
    finally:
        proc.stderr.close()
        if proc.poll() is None:
            proc.terminate()
    print(f"Saved {output_path} as ProRes 4444 with NCLC metadata and pixel-perfect blending")

def extract_frames_to_png(video_path, output_dir, is_mask=False):
    """Extract video to lossless PNG frames using FFmpeg"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Base command for lossless PNG extraction
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'image2',
        '-c:v', 'png',
        '-q:v', '0',
        '-bitexact'
    ]
    
    pattern = 'mask_%06d.png' if is_mask else 'frame_%06d.png'
    
    if is_mask:
        cmd += ['-pix_fmt', 'gray']
    else:
        if os.path.isfile(video_path):
            color_range, color_space, color_transfer, color_primaries, _, _, _ = probe_video_info(video_path)
        else:
            color_range, color_space, color_transfer, color_primaries = 'tv', 'bt709', 'bt709', 'bt709'
        
        cmd += ['-vf', f'scale=in_range={color_range}:out_range=full,format=rgb24']
        cmd += ['-colorspace', color_space, '-color_trc', color_transfer, '-color_primaries', color_primaries]
    
    cmd += [str(output_dir / pattern)]
    cmd += ['-y']  # Overwrite
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode != 0:
        print(f"FFmpeg extraction error: {result.stderr}")
        raise RuntimeError("Frame extraction failed")
    
    extracted_files = list(output_dir.glob('*.png'))
    return len(extracted_files)

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size

def read_frame_from_videos(frame_root):
    if os.path.isdir(frame_root):
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
    else:
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
    size = frames[0].size

    return frames, size
  
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if os.path.isdir(mpath):
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
    elif mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:
        # Extract from video
        temp_mask_dir = tempfile.mkdtemp()
        extract_frames_to_png(mpath, temp_mask_dir, is_mask=True)
        mnames = sorted(os.listdir(temp_mask_dir))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(temp_mask_dir, mp)))
          
    th = 127  # Adjusted threshold for binarization
    
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))
        
        # Binarize the mask first with the adjusted threshold
        bin_mask = (mask_img > th).astype(np.uint8)

        # For flow masks
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(bin_mask, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = bin_mask
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        # For dilated masks
        if mask_dilates > 0:
            mask_img_dil = scipy.ndimage.binary_dilation(bin_mask, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img_dil = bin_mask
        masks_dilated.append(Image.fromarray(mask_img_dil * 255))
    
    if len(masks_img) ==1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def load_models(device, use_half=False):
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    if use_half:
        fix_flow_complete = fix_flow_complete.half()
    fix_flow_complete.eval()

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path)
    model.to(device)
    if use_half:
        model = model.half()
    model.eval()
    
    return fix_raft, fix_flow_complete, model

def run_inference(video, mask, output, resize_ratio=1.0, height=-1, width=-1, mask_dilation=4, ref_stride=10, 
                  neighbor_length=10, subvideo_length=80, raft_iter=20, fp16=False, save_masked_in=False, models=None):
    device = get_device()
    use_half = fp16
    if device == torch.device('cpu'):
        use_half = False
    
    if use_half:
        print("Warning: fp16 is enabled, which may introduce minor pixel differences due to precision loss. For zero quality loss, run without fp16.")
    
    fix_raft, fix_flow_complete, model = models
    
    # If input is video, extract to PNG folders
    input_is_video = not os.path.isdir(video)
    mask_is_video = not os.path.isdir(mask)

    temp_dir = tempfile.mkdtemp()
    input_frames_dir = video
    mask_frames_dir = mask
    fps = 24.0
    if input_is_video:
        _, _, _, _, fps, _, _ = probe_video_info(video)
        input_frames_dir = os.path.join(temp_dir, 'input_frames')
        os.makedirs(input_frames_dir, exist_ok=True)
        extract_frames_to_png(video, input_frames_dir, is_mask=False)
    
    if mask_is_video:
        mask_frames_dir = os.path.join(temp_dir, 'mask_frames')
        os.makedirs(mask_frames_dir, exist_ok=True)
        extract_frames_to_png(mask, mask_frames_dir, is_mask=True)

    frames, size = read_frame_from_videos(input_frames_dir)

    if width != -1 and height != -1:
        size = (width, height)
    if resize_ratio != 1.0:
        size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)
    
    save_root = os.path.dirname(output)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    inpaint_out_path = output
    masked_in_path = os.path.splitext(output)[0] + '_masked.mov'

    frames_len = len(frames)
    print(f"Extracted {frames_len} frames from input.")

    flow_masks, masks_dilated = read_mask(mask_frames_dir, frames_len, size, 
                                          flow_mask_dilates=mask_dilation,
                                          mask_dilates=mask_dilation)
    masks_len = len(masks_dilated)
    print(f"Extracted {masks_len} masks from input.")

    if masks_len != frames_len and masks_len != 1:
        if masks_len > frames_len:
            print(f"Warning: Truncating masks from {masks_len} to {frames_len} to match frames.")
            flow_masks = flow_masks[:frames_len]
            masks_dilated = masks_dilated[:frames_len]
        else:
            raise ValueError(f"Mask count ({masks_len}) must match frame count ({frames_len}) or be 1 (static mask). Provide matching inputs.")

    w, h = size
    
    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
    
    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    print(f'\nProcessing [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=raft_iter)
            torch.cuda.empty_cache()

        if use_half:
            frames = frames.half()
            flow_masks = flow_masks.half()
            masks_dilated = masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
        
        # ---- complete flow ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            
        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, subvideo_length)  # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            
    
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1
    
    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            l_t = len(neighbor_ids)
            
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = (comp_frames[idx].astype(np.uint16) + img.astype(np.uint16)) // 2
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        torch.cuda.empty_cache()

    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

    # --- At final save ---
    comp_frames_uint8 = [np.clip(f, 0, 255).astype(np.uint8) for f in comp_frames]
    frames_inp_uint8 = [np.clip(f, 0, 255).astype(np.uint8) for f in frames_inp]
    masks_dilated_uint8 = [(masks_dilated[0, i, 0].cpu().numpy() > 0).astype(np.uint8) for i in range(video_length)]
    
    assert len(comp_frames_uint8) == len(frames_inp_uint8) == len(masks_dilated_uint8)
    
    save_video_highest_quality(
        frames=comp_frames_uint8,
        output_path=inpaint_out_path,
        fps=fps,
        reference_video=video,
        original_frames=frames_inp_uint8,
        masks=masks_dilated_uint8
    )
    
    # Masked preview
    if save_masked_in:
        masked_frames = []
        for i in range(video_length):
            mask_bool = masks_dilated[0, i, 0].cpu().numpy() > 0
            img = frames_inp[i].astype(np.float32)
            green = np.zeros_like(img)
            green[:,:,1] = 255.0
            alpha = 0.6
            fuse_img = (1-alpha)*img + alpha*green
            fuse_img[mask_bool] = fuse_img[mask_bool]
            masked_frames.append(fuse_img.clip(0,255).astype(np.uint8))
        save_video_highest_quality(masked_frames, masked_in_path, fps=fps, reference_video=video)
    
    print(f'\nAll results are saved in {save_root}')
    
    torch.cuda.empty_cache()

    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument('-o', '--output', type=str, default='results/output.mov', help='Path to the output video file.')
    parser.add_argument("--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument('--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument('--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument('--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument("--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument("--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument("--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument("--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')
    parser.add_argument('--save_masked_in', action='store_true', help='Save the masked input video.')

    args = parser.parse_args()

    device = get_device()
    models = load_models(device, args.fp16)
    
    run_inference(
        video=args.video,
        mask=args.mask,
        output=args.output,
        resize_ratio=args.resize_ratio,
        height=args.height,
        width=args.width,
        mask_dilation=args.mask_dilation,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        raft_iter=args.raft_iter,
        fp16=args.fp16,
        save_masked_in=args.save_masked_in,
        models=models
    )
