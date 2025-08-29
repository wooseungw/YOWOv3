import os
# Enable CPU fallback for unsupported MPS ops (e.g., max_pool3d)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scripts import train, ava_eval, ucf_eval, detect, live, onnx
import argparse
from utils.build_config import build_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOWOv3")

    parser.add_argument('-m', '--mode', type=str, help='train/eval/live/detect/onnx')
    parser.add_argument('-cf', '--config', type=str, help='path to config file')
    # live-specific source arg removed per user request

    args = parser.parse_args()

    config = build_config(args.config)

    if args.mode == 'train':
        train.train_model(config=config)

    elif args.mode == 'eval':
        if config['dataset'] == 'ucf' or config['dataset'] == 'jhmdb' or config['dataset'] == 'ucfcrime':
            ucf_eval.eval(config=config)
        elif config['dataset'] == 'ava':
            ava_eval.eval(config=config)

    elif args.mode == 'detect':
        detect.detect(config=config)

    elif args.mode == 'live':
        live.detect(config=config)
    
    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)
