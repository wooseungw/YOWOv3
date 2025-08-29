import os
import cv2
import torch
import argparse
import torchvision.transforms.functional as FT
from collections import deque
from PIL import Image

from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3
from utils.box import draw_bounding_box, non_max_suppression


# Enable CPU fallback for unsupported MPS ops (e.g., max_pool3d)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class FrameTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, bgr_frame):
        # Convert BGR numpy array to RGB PIL Image
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize([self.img_size, self.img_size])
        tensor = FT.to_tensor(img)
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_video(config, source, output, conf_thres=0.4, iou_thres=0.5):
    device = select_device()

    # Build model
    model = build_yowov3(config)
    # Ensure float32 BEFORE moving to MPS
    model = model.to(dtype=torch.float32).to(device)
    model.eval()

    img_size = config['img_size']
    clip_len = config.get('clip_length', 16)
    label_map = config['idx2name']

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")

    # Prepare writer
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if src_fps and src_fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (img_size, img_size))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {output}")

    transform = FrameTransform(img_size)
    window = deque(maxlen=clip_len)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # For visualization we will draw on resized frame later
            window.append(transform(frame))
            if len(window) < clip_len:
                continue

            # Build input clip [1, 3, T, H, W]
            clip = torch.stack(list(window), 0).permute(1, 0, 2, 3).contiguous()
            clip = clip.unsqueeze(0).to(device=device, dtype=torch.float32)

            # Forward (move outputs to CPU on MPS to run NMS)
            if device.type == 'mps':
                outputs = model(clip).to('cpu')
            else:
                outputs = model(clip)

            detections = non_max_suppression(outputs, conf_threshold=conf_thres, iou_threshold=iou_thres)[0]

            # Prepare frame for drawing (resize to model input size)
            vis = cv2.resize(frame, (img_size, img_size))
            if detections is not None and len(detections) > 0:
                draw_bounding_box(vis, detections[:, :4], detections[:, 5], detections[:, 4], label_map)

            writer.write(vis)
    finally:
        cap.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser(description='YOWOv3 video inference')
    parser.add_argument('-cf', '--config', required=True, type=str, help='path to config.yaml')
    parser.add_argument('--source', required=True, type=str, help='input video file (e.g., input.mp4)')
    parser.add_argument('--output', required=False, type=str, default='output.mp4', help='output video file')
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    args = parser.parse_args()

    config = build_config(args.config)
    infer_video(config, source=args.source, output=args.output, conf_thres=args.conf, iou_thres=args.iou)


if __name__ == '__main__':
    main()

