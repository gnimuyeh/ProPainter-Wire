# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device
import warnings
import av
import OpenEXR
import Imath
import subprocess
warnings.filterwarnings("ignore")
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
# Mapping for chromaticities based on FFmpeg AVColorPrimaries integer values
primaries_dict = {
    1: Imath.Chromaticities(Imath.V2f(0.64, 0.33), Imath.V2f(0.3, 0.6), Imath.V2f(0.15, 0.06), Imath.V2f(0.3127, 0.329)), # BT709
    9: Imath.Chromaticities(Imath.V2f(0.708, 0.292), Imath.V2f(0.170, 0.797), Imath.V2f(0.131, 0.046), Imath.V2f(0.3127, 0.329)), # BT2020
    6: Imath.Chromaticities(Imath.V2f(0.630, 0.340), Imath.V2f(0.310, 0.595), Imath.V2f(0.155, 0.070), Imath.V2f(0.3127, 0.3290)), # SMPTE170M
    5: Imath.Chromaticities(Imath.V2f(0.640, 0.330), Imath.V2f(0.290, 0.600), Imath.V2f(0.150, 0.060), Imath.V2f(0.3127, 0.3290)), # BT470BG
    7: Imath.Chromaticities(Imath.V2f(0.630, 0.340), Imath.V2f(0.310, 0.595), Imath.V2f(0.155, 0.070), Imath.V2f(0.3127, 0.3290)), # SMPTE240M
    8: Imath.Chromaticities(Imath.V2f(0.681, 0.319), Imath.V2f(0.243, 0.692), Imath.V2f(0.145, 0.049), Imath.V2f(0.3167, 0.3345)), # FILM
    10: Imath.Chromaticities(Imath.V2f(0.7347, 0.2653), Imath.V2f(0.0, 1.0), Imath.V2f(0.0001, -0.0770), Imath.V2f(0.32168, 0.33767)), # SMPTEST428
    11: Imath.Chromaticities(Imath.V2f(0.680, 0.320), Imath.V2f(0.265, 0.690), Imath.V2f(0.150, 0.060), Imath.V2f(0.314, 0.351)), # SMPTE431
    12: Imath.Chromaticities(Imath.V2f(0.680, 0.320), Imath.V2f(0.265, 0.690), Imath.V2f(0.150, 0.060), Imath.V2f(0.3127, 0.3290)), # SMPTE432
    22: Imath.Chromaticities(Imath.V2f(0.630, 0.340), Imath.V2f(0.295, 0.605), Imath.V2f(0.155, 0.077), Imath.V2f(0.3127, 0.3290)), # EBU3213
    4: Imath.Chromaticities(Imath.V2f(0.67, 0.33), Imath.V2f(0.21, 0.71), Imath.V2f(0.14, 0.08), Imath.V2f(0.310, 0.316)), # BT470M (added for completeness, as it's in FFmpeg enum)
}
primaries_names = {
    1: 'bt709',
    4: 'bt470m',
    5: 'bt470bg',
    6: 'smpte170m',
    7: 'smpte240m',
    8: 'film',
    9: 'bt2020',
    10: 'smpte428',
    11: 'smpte431',
    12: 'smpte432',
    22: 'jedec-p22',
}
# Name mappings for custom headers (based on FFmpeg enums)
color_space_names = { # AVColorSpace
    0: 'rgb',
    1: 'bt709',
    4: 'fcc',
    5: 'bt470bg',
    6: 'smpte170m',
    7: 'smpte240m',
    8: 'ycgco',
    9: 'bt2020nc',
    10: 'bt2020c',
    11: 'smpte2085',
    12: 'chroma-derived-nc',
    13: 'chroma-derived-c',
    14: 'ictcp',
}
transfer_names = { # AVColorTransferCharacteristic
    1: 'bt709',
    4: 'gamma22',
    5: 'gamma28',
    6: 'smpte170m',
    7: 'smpte240m',
    8: 'linear',
    9: 'log100',
    10: 'log316',
    11: 'iec61966-2-4',
    12: 'bt1361e',
    13: 'iec61966-2-1',
    14: 'bt2020-10',
    15: 'bt2020-12',
    16: 'smpte2084',
    17: 'smpte428',
    18: 'arib-std-b67',
}
range_names = { # AVColorRange
    0: 'tv',
    1: 'tv',
    2: 'pc',
}
def get_chromaticities(color_primaries):
    if color_primaries in primaries_dict:
        return primaries_dict[color_primaries]
    else:
        print(f"Unknown or unsupported color primaries {color_primaries}, falling back to BT.709")
        return primaries_dict[1]
def imwrite(img, file_path, params=None, auto_mkdir=True, color_info=None):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    # Save uint16 RGB as EXR using OpenEXR
    height, width, _ = img.shape
    header = OpenEXR.Header(width, height)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                          'G': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                          'B': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    if color_info and color_info['primaries'] is not None:
        header['chromaticities'] = get_chromaticities(color_info['primaries'])
    else:
        header['chromaticities'] = primaries_dict[1] # Default to BT.709
    header['whiteLuminance'] = 1.0
    exr = OpenEXR.OutputFile(file_path, header)
    img_half = (img / 65535.0).astype(np.float16)
    expected_size = width * height * 2
    r = img_half[:, :, 0].tobytes()
    g = img_half[:, :, 1].tobytes()
    b = img_half[:, :, 2].tobytes()
    if len(b) != expected_size:
        raise ValueError(f"Channel size mismatch: expected {expected_size}, got {len(b)}")
    exr.writePixels({'R': r, 'G': g, 'B': b})
    exr.close()
def save_video_highest_quality(frames, output_path, fps, color_info=None):
    if not frames:
        raise ValueError("No frames to save")
    h, w, _ = frames[0].shape
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb48le',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'prores_ks',
        '-profile:v', '5',
        '-pix_fmt', 'yuv444p12le',
        '-vendor', 'apl0',
        '-bits_per_mb', '8000',
        '-quant_mat', 'hq',
        '-movflags', '+write_colr+faststart',
        '-write_tmcd', '0',
        output_path
    ]
    primaries_name = primaries_names.get(color_info['primaries'], 'bt709') if color_info else 'bt709'
    matrix_name = color_space_names.get(color_info['matrix'], 'bt709') if color_info else 'bt709'
    transfer_name = transfer_names.get(color_info['transfer'], 'bt709') if color_info else 'bt709'
    range_name = range_names.get(color_info['range'], 'tv') if color_info else 'tv'
    cmd.insert(-1, '-color_primaries')
    cmd.insert(-1, primaries_name)
    cmd.insert(-1, '-colorspace')
    cmd.insert(-1, matrix_name)
    cmd.insert(-1, '-color_trc')
    cmd.insert(-1, transfer_name)
    cmd.insert(-1, '-color_range')
    cmd.insert(-1, range_name)
    print(f"Encoding to ProRes 4444 XQ (max quality mode) via direct pipe...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in tqdm(frames):
            proc.stdin.write(frame.astype(np.uint16).tobytes())
        proc.stdin.close()
        returncode = proc.wait()
        if returncode != 0:
            stderr = proc.stderr.read().decode()
            print(f"FFmpeg error: {stderr}")
            raise RuntimeError("FFmpeg encoding failed")
    finally:
        proc.stderr.close()
        if proc.poll() is None:
            proc.terminate()
    print(f"Saved {output_path} with maximum quality")
def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [cv2.resize(f, process_size, interpolation=cv2.INTER_CUBIC) for f in frames]
    else:
        out_size = frames[0].shape[1], frames[0].shape[0]
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [cv2.resize(f, process_size, interpolation=cv2.INTER_CUBIC) for f in frames]
    return frames, process_size, out_size
def read_frame_from_videos(frame_root):
    color_info = {'primaries': None, 'matrix': None, 'transfer': None, 'range': None}
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
        container = av.open(frame_root)
        video_stream = container.streams.video[0]
        pix_fmt = video_stream.codec_context.pix_fmt
        if pix_fmt in ['yuv444p12le', 'yuv444p12be', 'rgb48le', 'rgb48be', 'rgba64le', 'rgba64be']:
            bit_depth = 12
        elif pix_fmt in ['yuv444p10le', 'yuv444p10be']:
            bit_depth = 10
        else:
            bit_depth = 8
            print(f"Warning: Could not detect bit depth for pix_fmt={pix_fmt}, assuming 8-bit")
        print(f"Detected pixel format: {pix_fmt}, Bit depth: {bit_depth}")
        color_info['primaries'] = video_stream.codec_context.color_primaries
        color_info['matrix'] = video_stream.codec_context.colorspace
        color_info['transfer'] = video_stream.codec_context.color_trc
        color_info['range'] = video_stream.codec_context.color_range
        if color_info['primaries'] is None:
            color_info['primaries'] = 1
            print("No color primaries detected, defaulting to BT.709")
        if color_info['range'] is None or color_info['range'] == 0:
            video_stream.codec_context.color_range = 1
            color_info['range'] = 1
            print("Color range unspecified; assuming limited range (MPEG).")
        print(f"Detected color primaries: {primaries_names.get(color_info['primaries'], 'unknown')} ({color_info['primaries']})")
        print(f"Detected color space/matrix: {color_space_names.get(color_info['matrix'], 'unknown')} ({color_info['matrix']})")
        print(f"Detected transfer: {transfer_names.get(color_info['transfer'], 'unknown')} ({color_info['transfer']})")
        print(f"Detected color range: {range_names.get(color_info['range'], 'unknown')} ({color_info['range']})")
        frames = []
        frames_pil = []
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format='rgb48be')
                frames.append(img)
                frames_pil.append(Image.fromarray((img / 256).astype(np.uint8), mode='RGB'))
        fps = float(video_stream.average_rate)
        size = (video_stream.width, video_stream.height)
        video_name = os.path.basename(frame_root)[:-4]
        container.close()
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        frames_pil = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr), cv2.IMREAD_UNCHANGED)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.dtype == np.uint8:
                frame = frame.astype(np.uint16) * 257
            frames.append(frame)
            frames_pil.append(Image.fromarray(frame if frame.dtype == np.uint8 else (frame / 256).astype(np.uint8), mode='RGB'))
        fps = None
        size = frames[0].shape[1], frames[0].shape[0]
        bit_depth = 8
        # For image folders, default to BT.709
        color_info['primaries'] = 1
        color_info['range'] = 2
        print("Image folder input: Defaulting color primaries to BT.709")
        print("Image folder input: Defaulting color range to full (JPEG).")
    return frames, frames_pil, fps, size, video_name, color_info
def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    is_video = mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI'))
    is_single_image = mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'))
    if is_single_image:
        masks_img = [Image.open(mpath)]
    elif is_video:
        container = av.open(mpath)
        video_stream = container.streams.video[0]
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format='rgb24') # Read as 8-bit RGB for masks
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                masks_img.append(Image.fromarray(img_gray, mode='L'))
        container.close()
        if len(masks_img) != length:
            raise ValueError(f"Mask video frame count ({len(masks_img)}) does not match input video ({length})")
    else:
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length
    return flow_masks, masks_dilated
def extrapolation(video_ori, scale):
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].shape[1], video_ori[0].shape[0]
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)
    frames = []
    for v in video_ori:
        frame = np.zeros((imgH_extr, imgW_extr, 3), dtype=np.uint16)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(frame)
    masks_dilated = []
    flow_masks = []
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones((imgH_extr, imgW_extr), dtype=np.uint8)
    mask[H_start+dilate_h: H_start+imgH-dilate_h,
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))
    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)
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
    fix_flow_complete.eval()
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()
    if use_half and device != torch.device('cpu'):
        fix_raft = fix_raft.half()
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
    return {'raft': fix_raft, 'flow_complete': fix_flow_complete, 'model': model}
def run_inference(video, mask, output='results', resize_ratio=1.0, height=-1, width=-1, mask_dilation=0,
                  ref_stride=10, neighbor_length=10, subvideo_length=10, raft_iter=50,
                  mode='video_inpainting', scale_h=1.0, scale_w=1.2, save_fps=25,
                  save_frames=True, fp16=False, save_masked_in=False, models=None, device=None):
    if device is None:
        device = get_device()
    use_half = fp16 and device != torch.device('cpu')
    if models is None:
        models = load_models(device, use_half=use_half)
    fix_raft = models['raft']
    fix_flow_complete = models['flow_complete']
    model = models['model']
    frames, frames_pil, fps, size, video_name, color_info = read_frame_from_videos(video)
    if width != -1 and height != -1:
        size = (width, height)
    if resize_ratio != 1.0:
        size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))
    frames, size, out_size = resize_frames(frames, size)
    fps = save_fps if fps is None else fps
    save_root = os.path.join(output)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    if mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask, frames_len, size,
                                              flow_mask_dilates=mask_dilation,
                                              mask_dilates=mask_dilation)
        w, h = size
    elif mode == 'video_outpainting':
        assert scale_h is not None and scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (scale_h, scale_w))
        w, h = size
    else:
        raise NotImplementedError
    masked_frame_for_save = None
    if not save_frames and save_masked_in:
        masked_frame_for_save = []
        for i in range(len(frames)):
            mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.
            img = frames[i].astype(np.float32) / 65535.0
            green = np.zeros([h, w, 3])
            green[:, :, 1] = 1.0
            alpha = 0.6
            fuse_img = (1-alpha) * img + alpha * green
            fuse_img = mask_ * fuse_img + (1-mask_) * img
            masked_frame_for_save.append(np.clip(fuse_img * 65535.0, 0, 65535).astype(np.uint16))
    frames_inp = frames
    frames = to_tensors()(frames_pil).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        with torch.inference_mode():
            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames[:, f-1:end_f], iters=raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    del flows_f, flows_b
                    torch.cuda.empty_cache()
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
                del gt_flows_f_list, gt_flows_b_list, gt_flows_f, gt_flows_b
            else:
                gt_flows_bi = fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()
            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = fix_flow_complete.half()
                model = model.half()
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
                    del pred_flows_bi_sub
                    torch.cuda.empty_cache()
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
                del pred_flows_f, pred_flows_b
            else:
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()
            del gt_flows_bi
            torch.cuda.empty_cache()
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length)
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
                    del pred_flows_bi_sub, prop_imgs_sub, updated_frames_sub, updated_masks_sub, updated_local_masks_sub
                    torch.cuda.empty_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                del prop_imgs, updated_local_masks
                torch.cuda.empty_cache()
            del masked_frames
            torch.cuda.empty_cache()
    ori_frames = frames_inp
    comp_frames = [None] * video_length
    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        with torch.no_grad():
            with torch.inference_mode():
                l_t = len(neighbor_ids)
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() # [0,1] float
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    ori_scaled = ori_frames[idx].astype(np.float32) / 65535.0 # [0,1]
                    img = pred_img[i] * binary_masks[i] + ori_scaled * (1 - binary_masks[i])
                    img = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(np.uint16)
        del selected_imgs, selected_masks, selected_update_masks, selected_pred_flows_bi, pred_img, binary_masks
        torch.cuda.empty_cache()
    del pred_flows_bi
    torch.cuda.empty_cache()
    if save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            img_save_root = os.path.join(save_root, str(idx).zfill(4)+'.exr')
            imwrite(f, img_save_root, color_info=color_info)
    else:
        comp_frames = [cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC) for f in comp_frames]
        save_video_highest_quality(comp_frames, os.path.join(save_root, 'inpaint_out.mov'), fps=fps, color_info=color_info)
        if save_masked_in:
            masked_frame_for_save = [cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC) for f in masked_frame_for_save]
            save_video_highest_quality(masked_frame_for_save, os.path.join(save_root, 'masked_in.mov'), fps=fps, color_info=color_info)
    del frames, flow_masks, masks_dilated, updated_frames, updated_masks, ori_frames
    torch.cuda.empty_cache()
    print(f'\nAll results are saved in {save_root}')
if __name__ == '__main__':
    device = get_device()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument("--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument('--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument('--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument('--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument("--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument("--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument("--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument("--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument('--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument('--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument('--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument('--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument('--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')
    args = parser.parse_args()
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
        mode=args.mode,
        scale_h=args.scale_h,
        scale_w=args.scale_w,
        save_fps=args.save_fps,
        save_frames=args.save_frames,
        fp16=args.fp16,
        save_masked_in=True, # Default True for single run
        models=None,
        device=device
    )
