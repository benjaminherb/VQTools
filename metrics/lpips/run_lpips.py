import argparse
import os
import subprocess
import tempfile
import json 
import time
import lpips
import re
import numpy as np


def extract_frames(video_path, output_dir):
    cmd = ['ffmpeg', '-i', video_path, '-y', os.path.join(output_dir, 'frame_%06d.png')]
    subprocess.run(cmd, check=True, capture_output=True)

def run_lpips(reference, distorted, output_dir = '.', version='0.1', use_gpu=False):
    print(f"Running LPIPS {version} on {os.path.basename(distorted)} (REF: {os.path.basename(reference)})")
    loss_fn = lpips.LPIPS(net='alex', version=version)
    if use_gpu:
        loss_fn.cuda()

    output_file = f'{os.path.basename(distorted[:-4])}.lpips.json'
    output_file = os.path.join(output_dir, output_file)
    if os.path.exists(output_file):
        print("Skipping!")
        return

    with tempfile.TemporaryDirectory(dir='/run/media/ben/VSR/temp_lpips') as temp_dir:
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
                if i % (len(files) // 10) == 0:
                    print(f"{i / len(files) * 100}% finished ({i}/{len(files)})")
                frame_distances.append(float(dist01))
        
        results["frame_distances"] = frame_distances
        results["metadata"]["num_frames"] = len(frame_distances)
        results["metadata"]["mean_distance"] = np.mean(frame_distances)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)