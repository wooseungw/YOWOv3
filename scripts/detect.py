
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt

from cus_datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3 
from utils.build_config import build_config
from utils.flops import get_info

def detect(config):

    #########################################################################
    dataset = build_dataset(config, phase='test')
    model   = build_yowov3(config)
    # Show FLOPs/Params on CPU (safer for MPS)
    try:
        get_info(config, model)
    except Exception as e:
        print(f"[warn] get_info skipped: {e}")
    ##########################################################################
    # Select device: CUDA -> MPS -> CPU
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    mapping = config['idx2name']
    # Ensure float32 first, then move to device (for MPS compatibility)
    model = model.to(dtype=torch.float32 if device.type == 'mps' else torch.float64).to(device)
    model.eval()


    for idx in range(dataset.__len__()):
        origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)
        #print(bboxes)

        clip = clip.unsqueeze(0).to(device=device, dtype=torch.float32 if device.type == 'mps' else torch.float64)
        # Run model; if on MPS, move outputs to CPU for NMS ops
        if device.type == 'mps':
            outputs = model(clip).to('cpu')
        else:
            outputs = model(clip)
        outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]

        origin_image = cv2.resize(origin_image, (config['img_size'], config['img_size']))

        draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

        flag = 1 
        if flag:
            cv2.imshow('img', origin_image)
            k = cv2.waitKey(100)
            if k == ord('q'):
                return
        else:
            cv2.imwrite(r"H:\detect_images\_" + str(idx) + r".jpg", origin_image)

            print("ok")
            print("image {} saved!".format(idx))

if __name__ == "__main__":
    config = build_config()
    detect(config)
