#!/usr/bin/env python3
import argparse
import os
import yaml
import cv2
import torch
import numpy as np
from PIL import Image

from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3
from utils.box import non_max_suppression, draw_bounding_box


def to_tensor_clip(frames, img_size):
    # frames: list of numpy RGB HxWx3 in range 0..255
    import torchvision.transforms.functional as F
    pil_frames = [Image.fromarray(f).resize((img_size, img_size)).convert('RGB') for f in frames]
    tensors = [F.to_tensor(p) for p in pil_frames]
    clip = torch.stack(tensors, dim=1)  # [C, D, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    clip = (clip - mean) / std
    return clip


def load_idx2name_from_config(config):
    # Try label_map path first
    path = config.get('label_map') or config.get('labelmap')
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if 'idx2name' in data:
            # normalize keys to int
            out = {}
            for k, v in data['idx2name'].items():
                try:
                    out[int(k)] = v
                except Exception:
                    continue
            return out
        if 'action_id2name' in data:
            # Expect A001->name; convert to index 0-based
            out = {}
            for aid, v in data['action_id2name'].items():
                try:
                    if aid[0] in ('A', 'a') and aid[1:].isdigit():
                        out[int(aid[1:]) - 1] = v
                except Exception:
                    continue
            return out
    # fallback
    n = int(config.get('num_classes', 1))
    return {i: str(i) for i in range(n)}


def main():
    ap = argparse.ArgumentParser(description='Predict actions on a single MP4 video')
    ap.add_argument('--config', required=True, help='Path to config YAML (e.g., config/cf2/etri_config.yaml)')
    ap.add_argument('--video', required=True, help='Path to input MP4')
    ap.add_argument('--out', default='', help='Optional output MP4 path; if empty, just display')
    ap.add_argument('--stride', type=int, default=None, help='Frame step for buffering; if omitted, uses config sampling_rate')
    ap.add_argument('--keep-res', action='store_true', help='Draw boxes on original resolution and save at original size')
    args = ap.parse_args()

    config = build_config(args.config)
    clip_length = int(config['clip_length'])
    img_size = int(config['img_size'])
    conf_thres = float(config.get('detect_conf', 0.3))
    iou_thres = float(config.get('detect_iou', 0.5))

    # Build model
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    model = build_yowov3(config)
    # MPS must run in float32
    model = model.to(device=device, dtype=torch.float32 if device.type == 'mps' else torch.float32)
    model.eval()

    idx2name = load_idx2name_from_config(config)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_size = (W, H) if args.keep_res else (img_size, img_size)
        writer = cv2.VideoWriter(args.out, fourcc, fps, out_size)

    buffer = []
    fidx = 0
    stride = int(config.get('sampling_rate', 1)) if args.stride is None else max(1, int(args.stride))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        if (fidx - 1) % stride != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb)
        if len(buffer) > clip_length:
            buffer.pop(0)
        if len(buffer) < clip_length:
            continue

        clip = to_tensor_clip(buffer, img_size).unsqueeze(0)
        clip = clip.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out = model(clip)
            pred = out if device.type != 'mps' else out.to('cpu')
            det = non_max_suppression(pred, conf_threshold=conf_thres, iou_threshold=iou_thres)[0]

        # draw either on resized frame or original, and scale boxes if needed
        if args.keep_res:
            vis = frame.copy()
            if det.shape[0] > 0:
                boxes_draw = det[:, :4].clone()
                scale = torch.tensor([W / img_size, H / img_size, W / img_size, H / img_size], device=boxes_draw.device)
                boxes_draw = boxes_draw * scale
            else:
                boxes_draw = det[:, :4]
            draw_bounding_box(vis, boxes_draw, det[:, 5], det[:, 4], idx2name)
        else:
            vis = cv2.resize(frame, (img_size, img_size))
            draw_bounding_box(vis, det[:, :4], det[:, 5], det[:, 4], idx2name)

        if writer is not None:
            writer.write(vis)
        else:
            cv2.imshow('pred', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
