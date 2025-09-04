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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LPIPS on distorted videos.")
    parser.add_argument("-d", "--distorted", type=str, required=True, help="Path to the distorted video files.")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Path to the reference video files")
    parser.add_argument("-o", "--output_dir", type=str, default="./results", help="Directory to save the results.")
    parser.add_argument('--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

    args = parser.parse_args()

    for filename in os.listdir(args.distorted):
        #if not filename.endswith('.mkv') or 'decoded' in filename or filename.startswith('.'):
        if not filename.endswith('.mkv') or filename.startswith('.'):
            continue

        distorted_path = os.path.join(args.distorted, filename)
        reference_path = None

        match_default = re.match(r'^(.*)_(.*)_(\d+)x(\d+)_q(\d+)\.mkv$', filename)
        match_upscaled = re.match(r'^(.*)_(.*)_(\d+)x(\d+)_q(\d+)\..*\.mkv$', filename)
        if match_default:
            reference_path = os.path.join(args.reference, f"{match_default.group(1)}_original_3840x2160_q0.mkv")
        elif match_upscaled:
            reference_path = os.path.join(args.reference, f"{match_upscaled.group(1)}_original_3840x2160_q0.mkv")
        else:
            print(f"Skipping {distorted_path}")
            continue

        run_lpips(reference_path, distorted_path, args.output_dir)
