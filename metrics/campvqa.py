import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, print_separator, modify_file, create_venv, run_in_venv, is_quiet

REPO_URL = 'https://github.com/xinyiW915/CAMP-VQA.git'
CLIP_URL = 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'

# mode: (intra_cross_experiment, train_data_name, test_data_name)
# campvqa:         model trained on LSVQ (kfold)
# campvqa-finevd:  LSVQ model fine-tuned on FineVD (default of the official demo)
MODELS = {
    'campvqa': ('intra', 'lsvq_train', 'lsvq_test'),
    'campvqa-finevd': ('cross', 'lsvq_train', 'finevd'),
}

REQUIREMENTS = [
    'torch==2.6.0', 'torchvision==0.21.0', 'transformers==4.50.3', 'scikit-learn==1.6.1', 'joblib',
    'accelerate', 'huggingface-hub', 'safetensors', 'sentencepiece', 'tokenizers', 'timm', 'pytorchvideo',
    'fvcore', 'iopath', 'av', 'opencv-python', 'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
    'ffmpeg', 'ftfy', 'regex', 'tqdm', 'Pillow',
]


def check_campvqa(rebuild=False):
    """Clone the CAMP-VQA repo (includes the pretrained weights) and build its venv."""

    repo = Path(__file__).parent / "campvqa"

    if repo.exists() and rebuild:
        subprocess.run(['rm', '-rf', str(repo)], check=True)

    if not repo.exists():
        print_separator("BUILDING CAMP-VQA", newline=True)
        print_line("Cloning CAMP-VQA repository...", force=True)
        try:
            subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # quote the video path and fall back to the container bitrate (stream bitrate is missing for e.g. mkv)
            modify_file(str(repo / 'src' / 'camp-vqa_demo.py'), [
                {'action': 'replace', 'pattern': 'pix_fmt -of json {video_path}', 'content': 'pix_fmt:format=bit_rate -of json "{video_path}"'},
                {'action': 'replace', 'pattern': "bitrate = info['streams'][0].get('bit_rate', 0)", 'content': "bitrate = info['streams'][0].get('bit_rate', info.get('format', {}).get('bit_rate', 0))"},
            ])
            # use the actual video path instead of '<name>.mp4' (allows mkv/mov input)
            modify_file(str(repo / 'src' / 'extractor' / 'extract_frag.py'), [
                {'action': 'insert', 'pattern': 'filename = os.path.join(self.videos_dir, video_name)', 'content': [
                    "        if 'test_video_path' in row and pd.notnull(row['test_video_path']):",
                    "            filename = row['test_video_path']",
                ]},
            ])
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone CAMP-VQA repository: {e}", force=True)
            return False

    if not (repo / 'venv').exists():
        print_line("Creating CAMP-VQA virtual environment...")
        create_venv(str(repo / 'venv'), python='python3.10')
        run_in_venv(str(repo / 'venv'), ['pip', 'install', 'setuptools<81', 'wheel'])
        run_in_venv(str(repo / 'venv'), ['pip', 'install'] + REQUIREMENTS)
        run_in_venv(str(repo / 'venv'), ['pip', 'install', '--no-build-isolation', CLIP_URL])

    return True


def run_campvqa(mode, distorted, output_dir=None):
    """Run CAMP-VQA no-reference video quality assessment."""

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
        repo = Path(__file__).parent / "campvqa"
        intra_cross, train_data, test_data = MODELS[mode]
        device = 'gpu' if torch.cuda.is_available() else 'cpu'  # CAMP-VQA only supports cuda or cpu

        cmd = [
            'python', str(repo / 'src' / 'camp-vqa_demo.py'),
            '--device', device,
            '--intra_cross_experiment', intra_cross,
            '--train_data_name', train_data,
            '--test_data_name', test_data,
            '--test_video_path', os.path.abspath(distorted),
            '--save_model_path', str(repo / 'model'),
            '--prompt_path', str(repo / 'src' / 'config' / 'prompts.json'),
            '--num_workers', '0',
        ]
        result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(repo / 'src'))

        if result.returncode != 0:
            print_line(f"ERROR: CAMP-VQA evaluation failed: {result.stderr}", force=True)
            return None

        score = _parse_campvqa_results(result.stdout)
        if score is None:
            print_line(f"ERROR: Could not parse CAMP-VQA output: {result.stdout}", force=True)
            return None

        results = {
            'timestamp': ts(),
            'distorted': os.path.basename(distorted),
            'model': f"{train_data}->{test_data}",
            'score': score,
        }

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
        print_line(f"ERROR: CAMP-VQA evaluation failed: {e}", force=True)
        return None


def _parse_campvqa_results(stdout):
    """Parse the quality score from camp-vqa_demo.py output."""
    match = re.search(r'Predicted Quality Score:\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)', stdout)
    if match:
        return float(match.group(1))
    return None
