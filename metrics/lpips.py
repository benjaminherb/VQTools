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

from metrics.utils import get_output_filename, save_json


def extract_frames(video_path, output_dir):
    cmd = ['ffmpeg', '-i', video_path, '-y', os.path.join(output_dir, 'frame_%06d.png')]
    subprocess.run(cmd, check=True, capture_output=True)

def run_lpips(reference, distorted, mode, output_dir, net='alex', version='0.1', use_gpu=False):

    output_file = None
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print(f"{output_file} exists already - SKIPPING!")
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime.now()
    print("\nRESULTS")
    print(f"Start Time: {start_time}")

    # Suppress LPIPS setup messages
    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        loss_fn = lpips.LPIPS(net=net, version=version)
    
    if use_gpu:
        loss_fn.cuda()

    with tempfile.TemporaryDirectory() as temp_dir:
        ref_dir = os.path.join(temp_dir, 'reference')
        dist_dir = os.path.join(temp_dir, 'distorted')
        os.makedirs(ref_dir)
        os.makedirs(dist_dir)
        
        extract_frames(reference, ref_dir)
        extract_frames(distorted, dist_dir)
        
        results = {
            "metadata": {
                "reference_video": os.path.basename(reference),
                "distorted_video": os.path.basename(distorted),
                "name": os.path.basename(distorted)[:-4],
                "lpips_version": version,
                "gpu_used": use_gpu,
                "net": net,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
        }
        
        files = sorted(os.listdir(ref_dir))
        frame_distances = []
        for i, file in enumerate(files):
            if os.path.exists(os.path.join(dist_dir, file)):
                img0 = lpips.im2tensor(lpips.load_image(os.path.join(ref_dir, file)))
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(dist_dir, file)))
                
                if use_gpu:
                    img0 = img0.cuda()
                    img1 = img1.cuda()
                
                dist01 = loss_fn.forward(img0, img1)
                frame_distances.append(float(dist01.detach()))
        

        results["frame_distances"] = frame_distances
        results["metadata"]["num_frames"] = len(frame_distances)
        results["metadata"]["mean_distance"] = np.mean(frame_distances)

        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print(f"End Time:   {end_time}")
        print(f"Duration:   {analysis_duration}")
        
    if output_file is not None:
        save_json(results, output_file)
    
    print("LPIPS:     {:.4f}".format(results["metadata"]["mean_distance"]))
    return results 