import os
import subprocess
import tempfile
import urllib
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, check_docker, build_docker_image, print_line, create_venv, run_in_venv, print_separator, modify_file, get_device, is_quiet

MODEL_FILES = [
    ("COVER.pth", "https://github.com/vztu/COVER/raw/release/Model/COVER.pth"),
]

def check_cover(rebuild=False):
    """Check if COVER is available."""
    
    repo = Path(__file__).parent / "cover"
    if repo.exists() and rebuild:
        subprocess.run(['rm', '-rf', str(repo)], check=True)

    if not repo.exists():
        print_separator("BUILDING COVER", newline=True)
        print_line("Cloning COVER repository...", force=True)
        try:
            subprocess.run(['git', 'clone', 'https://github.com/taco-group/COVER.git', str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ## Set random seed for reproducible results and remove CUDA calls (issue in cpu-only environment)
            modify_file(str(repo / 'evaluate_one_video.py'), [
                {'action': 'insert', 'pattern': 'import torch', 'content': [
                    'import random',
                    'import numpy as np',
                    'torch.manual_seed(42)',
                    'np.random.seed(42)',
                    'random.seed(42)'
                ]},
                {'action': 'replace', 'pattern': 'torch.cuda.current_device()', 'content': ''},
                # Add device argument parser after line 36 (after other argument definitions)
                {'action': 'insert', 'line': 36, 'content': [
                    '    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")',
                ]},
                # Replace device detection with argument
                {'action': 'replace', 'pattern': 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")', 'content': 'device = torch.device(args.device)'}
            ])
            # handle mkv and mov files
            modify_file(str(repo / 'cover' / 'datasets' / 'cover_datasets.py'), [
                {'action': 'replace', 'pattern': 'elif video_path.endswith(".mp4"):', 'content': 'elif video_path.endswith((".mp4", ".mkv", ".mov")):'}
            ])

        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone Cover repository: {e}", force=True)
            return False

        if not (repo / 'venv').exists():
            print_line("Creating COVER virtual environment...")
            create_venv(str(repo / 'venv'), python='python3.8', requirements=str(repo / 'requirements.txt'), compile_decord=True)
            run_in_venv(str(repo / 'venv'), ['pip', 'install', 'pyiqa'])
            run_in_venv(str(repo / 'venv'), ['pip', 'install', '-e', str(repo)])

        weights_dir = repo / "pretrained_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        try:
            for name, url in MODEL_FILES:
                dest = weights_dir / name
                if dest.exists():
                    continue
                print_line(f"Downloading {name}...")
                urllib.request.urlretrieve(url, str(dest))

        except Exception as e:
            print_line(f"ERROR: Failed to download pretrained models: {e}", force=True)
            return False

        return True

    return True


def run_cover(mode, distorted, output_dir=None):
    """Run COVER video quality assessment."""
    
    # Prepare output file
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        temp_fd, output_file = tempfile.mkstemp(suffix='.json', prefix='dover_')
        os.close(temp_fd)
    
    start_time = datetime.now()
    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))
    
    try:
        repo_dir = Path(__file__).parent / "cover"
        device = get_device()
        device = "cpu" if device.type == "mps" else device  # COVER does not support mps
        cmd = [
            'python', str(repo_dir / 'evaluate_one_video.py'),
            '-v', os.path.abspath(distorted),
            '-d', str(device),
        ]
        result = run_in_venv(str(repo_dir / 'venv'), cmd, work_dir=str(repo_dir))
        
        if result.returncode != 0:
            print_line(f"ERROR: COVER evaluation failed: {result.stderr}", force=True)
            return None
        
        results = _parse_results(result.stdout, result.stderr, distorted)
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        
        if results:
            print_key_value("Semantic Score", f"{results['semantic_score']:.4f}")
            print_key_value("Technical Score", f"{results['technical_score']:.4f}")
            print_key_value("Aesthetic Score", f"{results['aesthetic_score']:.4f}")
            print_key_value("Overall Score", f"{results['fused_score']:.4f}")

            if is_quiet():
                print_line(f"COVER ({analysis_duration.total_seconds():.0f}s) | {results['fused_score']:.4f} | {os.path.basename(distorted)}", force=True)
        
        if output_file.endswith('.json'):
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: COVER evaluation failed: {e}", force=True)
        return None


def _parse_results(stdout, stderr, video_path=None):
    """Parse COVER results from evaluate_one_video output."""
    # Format: path, semantic score, technical score, aesthetic score, overall/final score
    # Example: video_1.mp4,-0.086216,-0.089703,0.105127,-0.070792
    
    lines = stdout.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('path,') and ',' in line:
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    filename = parts[0]
                    semantic_score = float(parts[1])
                    technical_score = float(parts[2])
                    aesthetic_score = float(parts[3])
                    overall_score = float(parts[4])
                    
                    return {
                        'timestamp': ts(),
                        'distorted': os.path.basename(video_path) if video_path else filename,
                        'semantic_score': semantic_score,
                        'technical_score': technical_score,
                        'aesthetic_score': aesthetic_score,
                        'fused_score': overall_score
                    }
                except (ValueError, IndexError):
                    continue
    
    return None