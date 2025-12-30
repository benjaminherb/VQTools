import argparse
import os
import subprocess
import tempfile
import json 
import time
import lpips
import re
import numpy as np
from datetime import datetime
import contextlib

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, extract_frames


def run_lpips(mode, distorted, reference, output_dir=None, version='0.1'):

    net = mode.split('-')[1]  # 'alex' or 'vgg'
    output_file = None
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime.now()
    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))

    # Suppress LPIPS setup messages
    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        loss_fn = lpips.LPIPS(net=net, version=version)
    
    device = get_device()
    loss_fn.to(device)

    dis_temp_dir = tempfile.TemporaryDirectory()
    ref_temp_dir = tempfile.TemporaryDirectory()
    distorted_frames = extract_frames(distorted, dis_temp_dir, fps=2)
    reference_frames = extract_frames(reference, ref_temp_dir, fps=2)
    
    if len(distorted_frames) != len(reference_frames):
        print_line(f"Frame count mismatch between distorted and reference videos.", force=True)
        return None
    
    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(distorted),
        "reference": os.path.basename(reference),
        "lpips_version": version,
        "device": str(device),
        "net": net,
        "fps": 2
    }
    
    frame_distances = []
    for dist_frame, ref_frame in zip(distorted_frames, reference_frames):
        img0 = lpips.im2tensor(lpips.load_image(ref_frame))
        img1 = lpips.im2tensor(lpips.load_image(dist_frame))

        img0 = img0.to(device)
        img1 = img1.to(device)

        dist01 = loss_fn.forward(img0, img1)
        frame_distances.append(float(dist01.detach()))
    
    results.update({
        f'lpips-{net}': np.mean(frame_distances),
        f'lpips-{net}_min': np.min(frame_distances),
        f'lpips-{net}_max': np.max(frame_distances),
        'num_frames': len(frame_distances),
        'frame_distances': frame_distances,
    })

    end_time = datetime.now()
    analysis_duration = end_time - start_time

    print_key_value("End Time", ts(end_time))
    print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        
    if output_file is not None:
        save_json(results, output_file)

    print_key_value("LPIPS", "{:.4f}".format(results["mean_distance"]), force=True)
    return results