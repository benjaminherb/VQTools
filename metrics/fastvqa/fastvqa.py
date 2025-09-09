import os
import re
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, get_device


MODEL_FILES = [
    ("FAST_VQA_B_1_4.pth", "https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth"),
    ("FAST_VQA_3D_1_1.pth", "https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth"),
]


def check_fastvqa():
    """Clone the FAST-VQA repo if missing and download pretrained weights if needed. """

    repo = Path(__file__).parent / "FAST-VQA-and-FasterVQA"
    if not repo.exists():
        print_line("Cloning FAST-VQA repository...", force=True)
        try:
            subprocess.run(['git', 'clone', 'https://github.com/VQAssessment/FAST-VQA-and-FasterVQA', str(repo)], check=True)
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone FastVQA repository: {e}", force=True)
            return False

    weights_dir = repo / "pretrained_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, url in MODEL_FILES:
            dest = weights_dir / name
            if dest.exists():
                continue
            print_line(f"Downloading {name}...")
            urllib.request.urlretrieve(url, str(dest))

            # create fallback symlinks for patterns like 1_4 -> 1*4 and 1_1 -> 1*1 (to avoid issues with unexpanded shell globbing)
            try: 
                for pat in ("1_4.pth", "1_1.pth"):
                    if not pat in name:
                        continue

                    alt_name = name.replace(pat, pat.replace("_", "*"))
                    alt_path = weights_dir / alt_name
                    print(f"Creating fallback symlink {alt_path} -> {dest.name}")
                    if not alt_path.exists():
                        print(f"Creating fallback symlink {alt_path} -> {dest.name}")
                        os.symlink(dest.name, str(alt_path))
            except Exception:
                pass
    except Exception as e:
        print_line(f"ERROR: Failed to download pretrained models: {e}", force=True)
        return False

    return True


def run_fastvqa(mode, distorted, output_dir=None):
    """Run FastVQA video quality assessment."""

    # Prepare output file
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
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(__file__).parent / "FAST-VQA-and-FasterVQA"

            if mode == 'fastvqa':
                mode_string = "FAST-VQA"
            elif mode == 'fastervqa':
                mode_string = "FasterVQA"
            else:
                return
            device = get_device()
            cmd = [
                'python3', str(work_dir / "vqa.py"),
                '-m', mode_string,
                '-v', os.path.abspath(distorted),
                '-d', str(device),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)

            if result.returncode != 0:
                print_line(f"ERROR: FastVQA evaluation failed: {result.stderr}", force=True)
                return None

            results = _parse_fastvqa_results(result.stdout, distorted)

            end_time = datetime.now()
            analysis_duration = end_time - start_time
            
            print_key_value("End Time", ts(end_time))
            print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
            if results:
                print_key_value("Score", results['score'])
            if output_file:
                save_json(results, output_file)
            
            return results
            
    except Exception as e:
        print_line(f"ERROR: FastVQA evaluation failed: {e}", force=True)
        return None

def _parse_fastvqa_results(stdout, video_path):
    """Parse FastVQA output for model, sampled frames, and score."""

    text = stdout if isinstance(stdout, str) else stdout.decode('utf-8', errors='ignore')

    model = re.search(r'Inferring with model\s*\[([^\]]+)\]', text, re.I).group(1).strip()
    score = re.search(r'quality score.*?is\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)', text, re.I).group(1)

    frames_string = re.search(r'Sampled frames are\s*\[([^\]]+)\]', text, re.I | re.S).group(1)
    sampled_frames = None
    if frames_string:
        nums = re.findall(r'\d+', frames_string)
        sampled_frames = [int(x) for x in nums]

    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(video_path),
        "model": model,
        "sampled_frames": sampled_frames,
        "score": score
    }

    return results