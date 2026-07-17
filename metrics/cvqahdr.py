import os
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, print_separator, modify_file, create_venv, run_in_venv, is_quiet

REPO_URL = 'https://github.com/sunwei925/CompressedVQA-HDR.git'

MODEL_LINKS = {
    "FR_HDR_VQA.pth": "https://www.dropbox.com/scl/fi/6745joi51g3fuubg2n87i/FR_HDR_VQA.pth?rlkey=atdpom6x6lmosk9tjqijbm44r&st=qcyio5gj&dl=1",
    "FR_HDR_VQA.npy": "https://www.dropbox.com/scl/fi/3po4q1e3ojfmvs6f83ow6/FR_HDR_VQA.npy?rlkey=pqb6jvzj2g2qdjpt7tnq8uueo&st=x9lqun6p&dl=1",
    "NR_HDR_VQA.pth": "https://www.dropbox.com/scl/fi/t9auox7p47yjf0crwxyso/NR_HDR_VQA.pth?rlkey=3hpgvoiq484lt80moh9j6tmkx&st=0q9ofeep&dl=1",
    "NR_HDR_VQA.npy": "https://www.dropbox.com/scl/fi/ezsv3buh353ny3y71y4wk/NR_HDR_VQA.npy?rlkey=domijnhpjfsjp9tyvvyfp5zf9&st=3u9734nt&dl=1",
}


def check_cvqahdr(rebuild=False):
    """Clone the CompressedVQA-HDR repo if missing, build its venv and check for the pretrained checkpoints."""

    repo = Path(__file__).parent / "cvqahdr"

    if repo.exists() and rebuild:
        subprocess.run(['rm', '-rf', str(repo)], check=True)

    if not repo.exists():
        print_separator("BUILDING CompressedVQA-HDR", newline=True)
        print_line("Cloning CompressedVQA-HDR repository...", force=True)
        try:
            subprocess.run(['git', 'clone', REPO_URL, str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            modify_file(str(repo / 'NR' / 'VQA_NR.py'), [
                {'action': 'replace', 'pattern': 'model.load_state_dict(torch.load(args.model_path))', 'content': 'model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))'},
                {'action': 'replace', 'pattern': '(video_read_index < video_length) and', 'content': '(video_read_index < video_length_read) and'}
            ])
            modify_file(str(repo / 'FR' / 'VQA_FR.py'), [
                {'action': 'replace', 'pattern': 'model.load_state_dict(torch.load(args.model_path))', 'content': 'model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))'},
                {'action': 'replace', 'pattern': '(video_read_index < video_length) and', 'content': '(video_read_index < video_length_read) and'}
            ])
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone CompressedVQA-HDR repository: {e}", force=True)
            return False

    ckpts_dir = repo / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, url in MODEL_LINKS.items():
            dest = ckpts_dir / name
            if dest.exists():
                continue
            print_line(f"Downloading {name}...")
            urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        print_line(f"ERROR: Failed to download pretrained models: {e}", force=True)
        print_line(f"Please download them manually and place them in {ckpts_dir}", force=True)
        for name, url in MODEL_LINKS.items():
            if not (ckpts_dir / name).exists():
                print_line(f"{name}: {url}", force=True)
        return False

    if not (repo / 'venv').exists():
        print_line("Creating CompressedVQA-HDR virtual environment...")
        create_venv(str(repo / 'venv'), python='python3.11')
        run_in_venv(str(repo / 'venv'), ['pip', 'install', 'numpy==1.26.4', 'opencv-python==4.11.0.86', 'Pillow==11.3.0', 'scipy==1.16.1', 'torch==2.4.1', 'torchvision==0.19.1', 'open_clip_torch'])

    return True


def run_cvqahdr(mode, distorted, reference, output_dir=None):
    """Run CompressedVQA-HDR (NR or FR) video quality assessment."""

    is_reference_based = 'cvqa-hdr-fr' in mode

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
    distorted = os.path.abspath(distorted)

    try:
        repo = Path(__file__).parent / "cvqahdr"

        if is_reference_based:
            reference = os.path.abspath(reference)
            cmd = [
                'python', str(repo / 'FR' / 'VQA_FR.py'),
                '--distorted', distorted,
                '--reference', reference,
                '--model_path', str(repo / 'ckpts' / 'FR_HDR_VQA.pth'),
                '--profile_path', str(repo / 'ckpts' / 'FR_HDR_VQA.npy'),
            ]
            work_dir = repo / 'FR'
        else:
            cmd = [
                'python', str(repo / 'NR' / 'VQA_NR.py'),
                '--distorted', distorted,
                '--model_path', str(repo / 'ckpts' / 'NR_HDR_VQA.pth'),
                '--profile_path', str(repo / 'ckpts' / 'NR_HDR_VQA.npy'),
            ]
            work_dir = repo / 'NR'

        result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(work_dir))

        if result.returncode != 0:
            print_line(f"ERROR: CompressedVQA-HDR evaluation failed: {result.stderr}", force=True)
            return None

        score = _parse_cvqahdr_results(result.stdout)
        if score is None:
            print_line(f"ERROR: Could not parse CompressedVQA-HDR output: {result.stdout}", force=True)
            return None

        results = {
            'timestamp': ts(),
            'distorted': os.path.basename(distorted),
            'score': score,
        }
        if is_reference_based:
            results['reference'] = os.path.basename(reference)

        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        print_key_value("Score", f"{score:.4f}")

        if is_quiet():
            print_line(f"{mode.upper()} ({analysis_duration.total_seconds():.0f}s) | {score:.4f} | {os.path.basename(distorted)}", force=True)

        if output_file:
            save_json(results, output_file)

        return results

    except Exception as e:
        print_line(f"ERROR: CompressedVQA-HDR evaluation failed: {e}", force=True)
        return None


def _parse_cvqahdr_results(stdout):
    """Parse the quality score from VQA_NR.py / VQA_FR.py output."""
    match = re.search(r'Quality score:\s*([0-9.]+)', stdout)
    if match:
        return float(match.group(1))
    return None
