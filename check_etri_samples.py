#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch

from utils.build_config import build_config
from cus_datasets.build_dataset import build_dataset


def unnormalize_frame(frame: torch.Tensor) -> np.ndarray:
    # frame: [C, H, W], normalized by mean/std as in transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = frame * std + mean
    img = (img.clamp(0, 1) * 255.0).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
    return img


def find_korean_font(preferred: str = "", size: int = 12):
    import os
    # Allow explicit path
    if preferred and os.path.isfile(preferred):
        try:
            return ImageFont.truetype(preferred, size)
        except Exception:
            pass

    # Common Korean-capable fonts across macOS/Linux/Windows
    candidates = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/NanumGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansKR-Regular.otf",
        "/Library/Fonts/NotoSansKR-Regular.otf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    # Fallback: default font (may not render Hangul)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Quick ETRI sample check and visualization")
    ap.add_argument('--config', default='config/cf2/etri_config.yaml')
    ap.add_argument('--participants', default='P02', help='P01 or comma separated, or all')
    ap.add_argument('--num', type=int, default=5, help='number of samples to check')
    ap.add_argument('--out', default='ava_result/etri_check', help='output folder to save images')
    ap.add_argument('--font', default='', help='path to a TTF/OTF font that supports Korean (optional)')
    ap.add_argument('--fontsize', type=int, default=12, help='label font size')
    ap.add_argument('--fps', type=float, default=None, help='Sample approx this many frames per second (e.g., 1.0)')
    args = ap.parse_args()

    cfg = build_config(args.config)
    cfg['dataset'] = 'etri'
    cfg['participants'] = args.participants

    # Use test phase to avoid random augmentations
    dataset = build_dataset(cfg, phase='test')

    os.makedirs(args.out, exist_ok=True)

    # Access dataset internals for verification
    # These attributes exist on the ETRI dataset class
    action2idx = getattr(dataset, 'action2idx', {})
    idx2action = getattr(dataset, 'idx2action', {})
    samples = getattr(dataset, 'samples', [])
    index = getattr(dataset, 'index', [])
    idx2name = cfg.get('idx2name', {})

    print(f"Dataset size (indices): {len(dataset)}")
    n = min(args.num, len(dataset))
    for i in range(n):
        i = i * 30
        # unravel to get metadata
        sample_idx, key_idx = index[i]
        rec = samples[sample_idx]
        base = rec['base']
        action_id = rec['action_id']
        expected_idx = action2idx[action_id]

        # optional font
        font = find_korean_font(args.font, size=args.fontsize)

        def draw_and_save(img_np, boxes, labels_np, suffix: str):
            H, W = img_np.shape[:2]
            pil = Image.fromarray(img_np).convert('RGBA')
            overlay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            ok_mapping = True
            for b, li in zip(boxes, labels_np):
                if isinstance(b, torch.Tensor):
                    b = b.cpu().numpy()
                x1 = int(b[0] * W); y1 = int(b[1] * H); x2 = int(b[2] * W); y2 = int(b[3] * H)
                name = idx2name.get(int(li), idx2action.get(int(li), str(li)))
                import colorsys
                hue = (int(li) * 37) % 360
                r, g, b = [int(255 * c) for c in colorsys.hsv_to_rgb(hue / 360.0, 0.9, 1.0)]
                color = (r, g, b, 255)
                for t in range(4):
                    draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
                text = f"{li}: {name}"
                try:
                    tb = draw.textbbox((0, 0), text, font=font)
                    tw = tb[2] - tb[0]
                    th = tb[3] - tb[1]
                except Exception:
                    try:
                        tw = draw.textlength(text, font=font)
                    except Exception:
                        tw = 10 * len(text)
                    th = (getattr(font, 'size', 16))
                pad = 6
                bg_w = int(tw) + pad * 2
                bg_h = int(th) + pad
                top_y = max(0, y1 - bg_h - 2)
                bg_x2 = min(W - 1, x1 + bg_w)
                bg_y2 = min(H - 1, top_y + bg_h)
                draw.rectangle([x1, top_y, bg_x2, bg_y2], fill=(0, 0, 0, 160))
                draw.text((x1 + pad, top_y + max(1, pad // 2)), text, fill=(255, 255, 255, 255), font=font)
                if int(li) != int(expected_idx):
                    ok_mapping = False
            out_path = os.path.join(args.out, f"sample_{i:03d}_{base}{suffix}.png")
            out_img = Image.alpha_composite(pil, overlay).convert('RGB')
            out_img.save(out_path)
            return ok_mapping, out_path

        if args.fps is None:
            # Use dataset pipeline (normalized frames)
            origin, clip, boxes, labels = dataset.__getitem__(i, get_origin_image=True)
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy().astype(int)
            else:
                labels_np = np.asarray(labels).astype(int)
            frame = clip[:, -1, :, :]
            img = unnormalize_frame(frame)
            ok_mapping, out_path = draw_and_save(img, boxes, labels_np, suffix="")
            label_names = [idx2name.get(int(li), idx2action.get(int(li), str(li))) for li in labels_np]
            print(f"[{i}] base={base} action_id={action_id} -> expected_idx={expected_idx}; labels={labels_np.tolist()} names={label_names}; bbox_n={len(boxes)}; match={'PASS' if ok_mapping else 'FAIL'}; saved={out_path}")
        else:
            # 1 FPS (or given) sampling from original video and CSV
            cap = cv2.VideoCapture(rec['rgb'])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps / args.fps)))
            total_rows = rec.get('length', 0)
            saved_count = 0
            from cus_datasets.etri.load_data import _parse_bbox_from_csv_row, _bi_size
            bi_size = _bi_size(rec['bi'])
            if bi_size is not None:
                W_bi, H_bi = bi_size
            else:
                W_bi = H_bi = None
            for fk in range(1, total_rows + 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fk - 1)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                img_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                H, W = img_np.shape[:2]
                # bbox from CSV
                row = dataset._load_csv_row(rec['csv'], fk)
                boxes = []
                if row is not None:
                    bbox = _parse_bbox_from_csv_row(row)
                    if bbox is not None:
                        x1_bi, y1_bi, x2_bi, y2_bi = bbox
                        if W_bi and H_bi and getattr(dataset, 'use_bodyindex_scale', True):
                            sx = W / max(1e-6, W_bi)
                            sy = H / max(1e-6, H_bi)
                            x1 = x1_bi * sx; y1 = y1_bi * sy; x2 = x2_bi * sx; y2 = y2_bi * sy
                        else:
                            x1, y1, x2, y2 = x1_bi, y1_bi, x2_bi, y2_bi
                        boxes = [np.array([x1 / W, y1 / H, x2 / W, y2 / H], dtype=np.float32)]
                labels_np = np.array([expected_idx], dtype=int)
                ok_mapping, out_path = draw_and_save(img_np, boxes, labels_np, suffix=f"_f{fk:05d}")
                saved_count += 1
            cap.release()
            print(f"[{i}] base={base} action_id={action_id} -> expected_idx={expected_idx}; sampled={saved_count} frames at ~{args.fps} FPS; saved to {args.out}")


if __name__ == '__main__':
    main()
