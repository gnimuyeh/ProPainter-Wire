# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import torch
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
import threading
import queue
import gc
import shutil

MODEL_PATH = "/runpod-volume/weights"

warnings.filterwarnings("ignore")
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

# --- Color Management & Constants ---
primaries_dict = {
    1: Imath.Chromaticities(Imath.V2f(0.64, 0.33), Imath.V2f(0.3, 0.6), Imath.V2f(0.15, 0.06), Imath.V2f(0.3127, 0.329)),
    9: Imath.Chromaticities(Imath.V2f(0.708, 0.292), Imath.V2f(0.170, 0.797), Imath.V2f(0.131, 0.046), Imath.V2f(0.3127, 0.329)),
}
primaries_names = {1: 'bt709', 4: 'bt470m', 5: 'bt470bg', 6: 'smpte170m', 7: 'smpte240m', 8: 'film', 9: 'bt2020', 10: 'smpte428', 11: 'smpte431', 12: 'smpte432', 22: 'jedec-p22'}
color_space_names = {0: 'rgb', 1: 'bt709', 4: 'fcc', 5: 'bt470bg', 6: 'smpte170m', 7: 'smpte240m', 8: 'ycgco', 9: 'bt2020nc', 10: 'bt2020c', 11: 'smpte2085', 12: 'chroma-derived-nc', 13: 'chroma-derived-c', 14: 'ictcp'}
transfer_names = {1: 'bt709', 4: 'gamma22', 5: 'gamma28', 6: 'smpte170m', 7: 'smpte240m', 8: 'linear', 9: 'log100', 10: 'log316', 11: 'iec61966-2-4', 12: 'bt1361e', 13: 'iec61966-2-1', 14: 'bt2020-10', 15: 'bt2020-12', 16: 'smpte2084', 17: 'smpte428', 18: 'arib-std-b67'}
range_names = {0: 'tv', 1: 'tv', 2: 'pc'}

def get_chromaticities(color_primaries):
    return primaries_dict.get(color_primaries, primaries_dict.get(1))

# --- I/O Helpers ---

def imwrite(img, file_path, color_info=None, auto_mkdir=True):
    if auto_mkdir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    height, width, _ = img.shape
    header = OpenEXR.Header(width, height)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                          'G': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                          'B': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    
    if color_info and color_info['primaries'] is not None:
        header['chromaticities'] = get_chromaticities(color_info['primaries'])
    else:
        header['chromaticities'] = get_chromaticities(1)

    header['whiteLuminance'] = 1.0
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)

    exr = OpenEXR.OutputFile(file_path, header)
    img_half = (img / 65535.0).astype(np.float16)
    
    exr.writePixels({'R': img_half[:, :, 0].tobytes(), 
                     'G': img_half[:, :, 1].tobytes(), 
                     'B': img_half[:, :, 2].tobytes()})
    exr.close()

class AsyncFrameWriter:
    def __init__(self, output_path, fps, color_info, save_frames=False, is_video=False):
        self.queue = queue.Queue(maxsize=30)
        self.output_path = output_path
        self.save_frames = save_frames
        self.color_info = color_info
        self.stop_event = threading.Event()
        self.proc = None
        self.is_video_mode = is_video and not save_frames
        
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def _init_ffmpeg(self, w, h, fps):
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-c:v', 'prores_ks', '-profile:v', '5', '-pix_fmt', 'yuv444p12le',
            '-vendor', 'apl0', '-bits_per_mb', '8000', '-quant_mat', 'hq',
            '-bitexact', '-movflags', '+write_colr+faststart',
            os.path.join(self.output_path, 'inpaint_out.mov')
        ]
        c = self.color_info
        if c:
            cmd.insert(-1, '-color_primaries'); cmd.insert(-1, primaries_names.get(c['primaries'], 'bt709'))
            cmd.insert(-1, '-colorspace'); cmd.insert(-1, color_space_names.get(c['matrix'], 'bt709'))
            cmd.insert(-1, '-color_trc'); cmd.insert(-1, transfer_names.get(c['transfer'], 'bt709'))
            cmd.insert(-1, '-color_range'); cmd.insert(-1, range_names.get(c['range'], 'tv'))

        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1)
                frame, idx, width, height, fps = item
                
                if self.save_frames:
                    imwrite(frame, os.path.join(self.output_path, str(idx).zfill(4)+'.exr'), 
                           color_info=self.color_info, auto_mkdir=(idx==0))
                else:
                    if self.proc is None:
                        self.proc = self._init_ffmpeg(width, height, fps)
                    self.proc.stdin.write(frame.tobytes())
                
                self.queue.task_done()
            except queue.Empty:
                continue

    def write(self, frame, idx=0, w=0, h=0, fps=24):
        self.queue.put((frame, idx, w, h, fps))

    def close(self):
        self.stop_event.set()
        self.thread.join()
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                print(f"FFmpeg Error: {self.proc.stderr.read().decode()}")
            self.proc.stderr.close()

# --- STREAMING COMPOSITOR ---
class StreamingCompositor:
    def __init__(self, writer, original_w, original_h, fps):
        self.writer = writer
        self.buffer = {} # Stores {idx: (img_sum, count)}
        self.last_written_idx = -1
        self.w = original_w
        self.h = original_h
        self.fps = fps

    def add_frames(self, frames_np, indices):
        """
        frames_np: List or Array of frames [N, H, W, 3] (uint16)
        indices: List of frame indices
        """
        for i, idx in enumerate(indices):
            img = frames_np[i].astype(np.float32)
            if idx in self.buffer:
                s, c = self.buffer[idx]
                self.buffer[idx] = (s + img, c + 1)
            else:
                self.buffer[idx] = (img, 1)

    def flush(self, up_to_idx):
        """
        Writes frames STRICTLY in order. 
        Will only write frame 'N' if 'N-1' has been written.
        """
        # We want to write everything in buffer <= up_to_idx
        # But we must do it sequentially: 0, 1, 2, ...
        
        while True:
            next_idx = self.last_written_idx + 1
            
            # If the next expected frame is in buffer AND is safe to write
            if next_idx in self.buffer and next_idx <= up_to_idx:
                s, c = self.buffer[next_idx]
                
                # Average
                final_img = (s / c)
                final_img = np.clip(final_img, 0, 65535).astype(np.uint16)
                
                # Crop
                final_img = final_img[:self.h, :self.w, :]
                
                self.writer.write(final_img, idx=next_idx, w=self.w, h=self.h, fps=self.fps)
                
                del self.buffer[next_idx]
                self.last_written_idx = next_idx
            else:
                # We either caught up to up_to_idx OR we are missing a frame (gap)
                break

    def finish(self):
        """Flush remaining buffer regardless of threshold."""
        self.flush(float('inf'))

def copy_metadata(source_path, target_path):
    temp_path = target_path + "_temp.mov"
    cmd = [
        'ffmpeg', '-y', '-i', target_path, '-i', source_path,
        '-map', '0:v', '-map_metadata', '1', '-map_metadata:s:v', '1:s:v',
        '-c', 'copy', '-movflags', 'use_metadata_tags', temp_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_path):
        os.replace(temp_path, target_path)

# --- Processing Helpers ---

def pad_frames_to_div8(frames, mask_frames=None):
    h, w = frames[0].shape[:2]
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    
    if pad_h == 0 and pad_w == 0:
        return frames, mask_frames, (w, h), (0, 0)
    
    print(f"Padding input {w}x{h} -> {w+pad_w}x{h+pad_h} (Reflection)")
    padded_frames = [cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT) for f in frames]
    
    padded_masks = None
    if mask_frames is not None:
        padded_masks = []
        for m in mask_frames:
            pm = cv2.copyMakeBorder(np.array(m), 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            padded_masks.append(Image.fromarray(pm))
            
    return padded_frames, padded_masks, (w + pad_w, h + pad_h), (pad_w, pad_h)

def read_frame_from_videos(frame_root):
    color_info = {'primaries': 1, 'matrix': 1, 'transfer': 1, 'range': 1}
    frames = []
    fps = 24.0
    video_name = os.path.basename(frame_root)

    if frame_root.endswith(('mp4', 'mov', 'avi', 'mkv')):
        video_name = video_name[:-4]
        container = av.open(frame_root)
        video_stream = container.streams.video[0]
        
        color_info['primaries'] = video_stream.codec_context.color_primaries or 1
        color_info['matrix'] = video_stream.codec_context.colorspace or 1
        color_info['transfer'] = video_stream.codec_context.color_trc or 1
        color_info['range'] = video_stream.codec_context.color_range or 1
        fps = float(video_stream.average_rate)
        
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format='rgb48be'))
        container.close()
    else:
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            if fr.startswith('.'): continue
            frame = cv2.imread(os.path.join(frame_root, fr), cv2.IMREAD_UNCHANGED)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.dtype == np.uint8:
                frame = frame.astype(np.uint16) * 257
            frames.append(frame)
        color_info['range'] = 2

    size = (frames[0].shape[1], frames[0].shape[0])
    return frames, fps, size, video_name, color_info

def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    
    if mpath.endswith(('mp4', 'mov', 'avi', 'mkv')):
        container = av.open(mpath)
        container.streams.video[0].thread_type = "AUTO"
        for packet in container.demux(container.streams.video[0]):
            for frame in packet.decode():
                img = frame.to_ndarray(format='rgb24')
                masks_img.append(Image.fromarray(img[:, :, 1])) # Green channel
        container.close()
    elif mpath.endswith(('jpg', 'png', 'exr', 'tga')):
        masks_img = [Image.open(mpath)]
    else:
        mnames = sorted(os.listdir(mpath))
        masks_img = [Image.open(os.path.join(mpath, m)) for m in mnames if not m.startswith('.')]

    current_len = len(masks_img)
    if current_len == 1:
        masks_img = masks_img * length
    elif current_len != length:
        if current_len > length: masks_img = masks_img[:length]
        else: masks_img.extend([masks_img[-1]] * (length - current_len))

    # DEBUG
    if len(masks_img) > 0:
        t = np.array(masks_img[0])
        print(f"[Mask Debug] Min: {t.min()} Max: {t.max()} (Safe Threshold > 127 applied)")

    flow_masks_pil, masks_dilated_pil = [], []
    
    for mask_img in masks_img:
        if size and mask_img.size != size:
            mask_img = mask_img.resize(size, Image.NEAREST)
        
        m_arr = np.array(mask_img.convert('L'))
        
        # --- FIXED THRESHOLD (> 127) ---
        binary = np.zeros_like(m_arr)
        binary[m_arr > 127] = 1 
        
        if flow_mask_dilates > 0:
            fm = scipy.ndimage.binary_dilation(binary, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            fm = binary
            
        if mask_dilates > 0:
            md = scipy.ndimage.binary_dilation(binary, iterations=mask_dilates).astype(np.uint8)
        else:
            md = binary
        
        flow_masks_pil.append(Image.fromarray(fm * 255))
        masks_dilated_pil.append(Image.fromarray(md * 255))
        
    return flow_masks_pil, masks_dilated_pil

def load_models(device, use_half=False):
    ckpt_path = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), model_dir=MODEL_PATH', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), model_dir=MODEL_PATH', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters(): p.requires_grad = False
    fix_flow_complete.to(device).eval()
    
    ckpt_path = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), model_dir=MODEL_PATH', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device).eval()
    
    if use_half:
        # fix_raft = fix_raft.half()  <-- DELETE or COMMENT OUT THIS LINE
        # RAFT must remain in Float32 for stability and to avoid the type error.
        
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()
    
    return {'raft': fix_raft, 'flow_complete': fix_flow_complete, 'model': model}

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids: ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num: break
                ref_index.append(i)
    return ref_index

# --- Main Inference Pipeline ---

def run_inference(video, mask, output='results', resize_ratio=1.0, height=-1, width=-1, mask_dilation=0,
                  ref_stride=10, neighbor_length=10, subvideo_length=10, raft_iter=50,
                  mode='video_inpainting', scale_h=1.0, scale_w=1.2, save_fps=25,
                  save_frames=True, fp16=False, save_masked_in=False, models=None, device=None):
    
    if device is None: device = get_device()
    use_half = fp16 and device != torch.device('cpu')
    if models is None: models = load_models(device, use_half=use_half)

    fix_raft, fix_flow_complete, model = models['raft'], models['flow_complete'], models['model']

    # 1. Load Data
    print(f"Loading {video}...")
    frames, fps, size, video_name, color_info = read_frame_from_videos(video)
    original_w, original_h = size
    fps = save_fps if fps is None else fps

    # 2. Pad
    if mode == 'video_inpainting':
        frames_len = len(frames)
        raw_flow_masks, raw_masks_dilated = read_mask(mask, frames_len, size, flow_mask_dilates=mask_dilation, mask_dilates=mask_dilation)
        frames, frames_pil_masks, padded_size, (pad_w, pad_h) = pad_frames_to_div8(frames, raw_masks_dilated)
        w, h = padded_size
        
        flow_masks_tensor = torch.stack([torch.from_numpy(np.array(m)).float() / 255.0 for m in frames_pil_masks]).unsqueeze(1)
        masks_dilated_tensor = torch.stack([torch.from_numpy(np.array(m)).float() / 255.0 for m in frames_pil_masks]).unsqueeze(1)
    
    # 3. Tensor Convert
    print("Converting to tensors...")
    frames_tensor_list = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 65535.0
        t = t * 2.0 - 1.0
        frames_tensor_list.append(t)
    
    frames_tensor = torch.stack(frames_tensor_list).unsqueeze(0)
    del frames, frames_tensor_list, raw_flow_masks, raw_masks_dilated
    gc.collect()

    frames_tensor = frames_tensor.to(device)
    flow_masks_tensor = flow_masks_tensor.unsqueeze(0).to(device)
    masks_dilated_tensor = masks_dilated_tensor.unsqueeze(0).to(device)

    if use_half:
        frames_tensor = frames_tensor.half()
        flow_masks_tensor = flow_masks_tensor.half()
        masks_dilated_tensor = masks_dilated_tensor.half()

    video_length = frames_tensor.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')

    # --- INFERENCE ---
    with torch.no_grad():
        with torch.inference_mode():
            if frames_tensor.size(-1) <= 640: short_clip_len = 12
            elif frames_tensor.size(-1) <= 720: short_clip_len = 8
            elif frames_tensor.size(-1) <= 1280: short_clip_len = 4
            else: short_clip_len = 2
            
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                # We cast the input slice to .float() because fix_raft is running in FP32
                if f == 0: 
                    flows_f, flows_b = fix_raft(frames_tensor[:, f:end_f].float(), iters=raft_iter)
                else: 
                    flows_f, flows_b = fix_raft(frames_tensor[:, f-1:end_f].float(), iters=raft_iter)
                gt_flows_f_list.append(flows_f); gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            
            gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
            del gt_flows_f_list, gt_flows_b_list
            if use_half: gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_tensor)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_tensor)
            del gt_flows_bi

            masked_frames = frames_tensor * (1 - masks_dilated_tensor)
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated_tensor, 'nearest')
            updated_frames = frames_tensor * (1 - masks_dilated_tensor) + prop_imgs.view(1, video_length, 3, h, w) * masks_dilated_tensor
            updated_masks = updated_local_masks.view(1, video_length, 1, h, w)
            del prop_imgs, updated_local_masks

    # --- COMPOSITION & ASYNC SAVING ---
    if not os.path.exists(output): os.makedirs(output, exist_ok=True)
    
    writer = AsyncFrameWriter(output, fps, color_info, save_frames=save_frames, is_video=not save_frames)
    compositor = StreamingCompositor(writer, original_w, original_h, fps)

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length: ref_num = subvideo_length // ref_stride
    else: ref_num = -1

    print("Compositing and Saving...")
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated_tensor[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        with torch.no_grad():
            with torch.inference_mode():
                l_t = len(neighbor_ids)
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                
                pred_img = pred_img.view(-1, 3, h, w) 
                pred_img = (pred_img + 1) / 2
                
                binary_masks = masks_dilated_tensor[0, neighbor_ids, :, :, :]
                ori_gpu = frames_tensor[0, neighbor_ids, :, :, :] 
                ori_gpu = (ori_gpu + 1) / 2 

                comp_tensor = pred_img * binary_masks + ori_gpu * (1 - binary_masks)
                comp_cpu = comp_tensor.permute(0, 2, 3, 1).cpu().numpy() 

                batch_frames = []
                for i in range(len(neighbor_ids)):
                    frame = np.clip(comp_cpu[i] * 65535.0, 0, 65535).astype(np.uint16)
                    batch_frames.append(frame)

                compositor.add_frames(batch_frames, neighbor_ids)
                
                # IMPORTANT: Flush up to the frame before the *start* of the current sliding window.
                # frames inside the window are still being blended.
                safe_threshold = max(0, f - neighbor_stride) - 1
                compositor.flush(safe_threshold)

        torch.cuda.empty_cache()

    compositor.finish()
    writer.close()
    
    if not save_frames and os.path.isfile(video):
        final_vid_path = os.path.join(output, 'inpaint_out.mov')
        copy_metadata(video, final_vid_path)
        
    print(f'\nAll results are saved in {output}')

if __name__ == '__main__':
    device = get_device()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees')
    parser.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask')
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument("--resize_ratio", type=float, default=1.0)
    parser.add_argument('--height', type=int, default=-1)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--mask_dilation', type=int, default=4)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=10)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--raft_iter", type=int, default=20)
    parser.add_argument('--mode', default='video_inpainting')
    parser.add_argument('--scale_h', type=float, default=1.0)
    parser.add_argument('--scale_w', type=float, default=1.2)
    parser.add_argument('--save_fps', type=int, default=24)
    parser.add_argument('--save_frames', action='store_true')
    parser.add_argument('--fp16', action='store_true')
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
        save_fps=args.save_fps,
        save_frames=args.save_frames,
        fp16=args.fp16,
        save_masked_in=True,
        device=device
    )
