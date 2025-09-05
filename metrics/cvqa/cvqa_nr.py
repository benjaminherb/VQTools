# Original work: https://github.com/sunwei925/CompressedVQA.git
# Paper: Deep Learning based Full-reference and No-reference Quality Assessment Models for Compressed UGC Videos
# https://doi.org/10.48550/arXiv.2106.01111
# Licensed under Apache License 2.0
# Copyright [2021, Wei Sun]

import argparse
import os
import json
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn
from metrics.cvqa import UGCVQA_NR_model

# Suppress torchvision deprecation warnings
warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*", category=UserWarning)
import cv2
from PIL import Image

from torchvision import transforms

def video_processing(dist):

    video_name = dist
    video_name_dis = video_name

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap=cv2.VideoCapture(video_name)

    video_channel = 3

    video_height_crop = 448
    video_width_crop = 448

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_length_read = int(video_length/video_frame_rate)

    transformations = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    transformed_video = torch.zeros([video_length_read, video_channel,  video_height_crop, video_width_crop])

    video_read_index = 0
    frame_idx = 0
            
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):

                read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                read_frame = transformations(read_frame)
                transformed_video[video_read_index] = read_frame
                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    video_capture.release()

    video = transformed_video


    return video, video_name_dis


def video_processing_multi_scale(dist):

    video_name = dist
    video_name_dis = video_name

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap=cv2.VideoCapture(video_name)


    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if video_height > video_width:
        video_width_resize = 540
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = 540
        video_width_resize = int(video_height_resize/video_height*video_width)

    dim1 = (video_height_resize, video_width_resize)

    if video_height > video_width:
        video_width_resize = 720
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = 720
        video_width_resize = int(video_height_resize/video_height*video_width)

    dim2 = (video_height_resize, video_width_resize)


    if video_height > video_width:
        video_width_resize = 1080
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = 1080
        video_width_resize = int(video_height_resize/video_height*video_width)

    dim3 = (video_height_resize, video_width_resize)


    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_length_read = int(video_length/video_frame_rate)

    transformed_video1 = torch.zeros([video_length_read, video_channel, dim1[0], dim1[1]])
    transformed_video2 = torch.zeros([video_length_read, video_channel, dim2[0], dim2[1]])
    transformed_video3 = torch.zeros([video_length_read, video_channel, dim3[0], dim3[1]])

    transformations1 = transforms.Compose([transforms.Resize(dim1), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transformations2 = transforms.Compose([transforms.Resize(dim2), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transformations3 = transforms.Compose([transforms.Resize(dim3), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])        

    video_read_index = 0
    frame_idx = 0
            
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):

                read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                read_frame1 = transformations1(read_frame)
                transformed_video1[video_read_index] = read_frame1

                read_frame2 = transformations2(read_frame)
                transformed_video2[video_read_index] = read_frame2

                read_frame3 = transformations3(read_frame)
                transformed_video3[video_read_index] = read_frame3
                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video1[i] = transformed_video1[video_read_index - 1]
            transformed_video2[i] = transformed_video2[video_read_index - 1]
            transformed_video3[i] = transformed_video3[video_read_index - 1]

    video_capture.release()
    video1= transformed_video1
    video2= transformed_video2
    video3 = transformed_video3


    return  video1, video2, video3, video_name_dis

def run_compressed_vqa_nr(dist, output, multiscale=False, is_gpu=False):

    device = torch.device('cuda' if is_gpu else 'cpu')
    # print('using ' + str(device))

    model = UGCVQA_NR_model.resnet50(pretrained=True)
    # model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model_path = os.path.join(os.path.dirname(__file__), 'ckpts', 'UGCVQA_NR_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if not multiscale:
        video_dist, video_name = video_processing(dist)

        with torch.no_grad():
            model.eval()

            video_dist = video_dist.to(device)

            video_dist = video_dist.unsqueeze(dim=0)

            outputs = model(video_dist)
            
            y_val = outputs.item()

    else: # multiscale
        video_dist1, video_dist2, video_dist3, video_name = video_processing_multi_scale(dist)

        with torch.no_grad():
            model.eval()

            video_dist1 = video_dist1.to(device)
            video_dist1 = video_dist1.unsqueeze(dim=0)
            outputs1 = model(video_dist1)           
            y_val1 = outputs1.item()

            video_dist2 = video_dist2.to(device)
            video_dist2 = video_dist2.unsqueeze(dim=0)
            outputs2 = model(video_dist2)            
            y_val2 = outputs2.item()

            video_dist3 = video_dist3.to(device)
            video_dist3 = video_dist3.unsqueeze(dim=0)
            outputs3 = model(video_dist3)         
            y_val3 = outputs3.item()

            w1_csf = 0.8317
            w2_csf = 0.0939
            w3_csf = 0.0745

            y_val = pow(y_val1, w1_csf) * pow(y_val2, w2_csf) * pow(y_val3, w3_csf)



    if not os.path.exists(output):
        os.system(r"touch {}".format(output))

    result = {
        "name": os.path.splitext(os.path.basename(video_name))[0],
        "distorted": os.path.basename(video_name),
        "model": "UGCVQA_NR",
        "multiscale": multiscale,
        "score": float(y_val),
        "ts": str(datetime.now())
    }
    
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dist', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--multiscale', action='store_true', default=False, help='Use multi-scale method')
    parser.add_argument('--is_gpu', action='store_true', default=False)
  

    config = parser.parse_args()

    run_compressed_vqa_nr(config.dist, config.output, config.multiscale, config.is_gpu)