
import os
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, print_separator, create_venv, run_in_venv, modify_file

def check_qalign(rebuild=False):
    """Check if QALIGN is available."""
    repo_path = Path(__file__).parent / "qalign"
    venv_path = repo_path / "venv"

    if rebuild and repo_path.exists():
        subprocess.run(['rm', '-rf', str(repo_path)], check=True)

    if not repo_path.exists():
        try:
            print_separator("BUILDING QALIGN", newline=True)
            print_line("Cloning QALIGN repository...")
            result = subprocess.run(['git', 'clone', 'https://github.com/q-future/q-align.git', repo_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

            # switch to 32 bit floats for CPU processing
            modify_file(repo_path / 'q_align' / 'evaluate' / 'scorer.py', [
                {'action': 'replace', 'pattern': 'self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)',
                 'content': 'self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).to(model.device) if device == "cpu" else torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)'},
                {'action': 'replace', 'pattern': 'video_tensors = [self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(self.model.device) for vid in video]',
                 'content': 'video_tensors = [self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].to(self.model.device) if self.model.device.type == "cpu" else self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(self.model.device) for vid in video]'},
            ])

            modify_file(repo_path / 'q_align' / 'model' / 'builder.py', [
                {'action': 'replace', 'pattern': "kwargs['torch_dtype'] = torch.float16", 'content': "kwargs['torch_dtype'] = torch.float16 if device != 'cpu' else torch.float32"},
            ])

            print("Creating virtual environment...")
            create_venv(str(venv_path), python='python3.8', compile_decord=True)
            run_in_venv(str(venv_path), ['pip', 'install', '-e', str(repo_path)])
            run_in_venv(str(venv_path), ['pip', 'install', 'numpy==1.24.3', 'protobuf'])


        except Exception as e:
            print_line(f"ERROR: Failed to build Q-Align - {e}", force=True)
            return False

    return True


def run_qalign(mode, distorted, output_dir=None):

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
    repo_path = Path(__file__).parent / "qalign"
    venv_path = repo_path / "venv"
    device = get_device()
    if str(device)== 'mps':
        device = 'cpu'  # MPS not supported, fallback to CPU
    result = run_in_venv(str(venv_path), ['python', 'q_align/evaluate/scorer.py', '--device', str(device), '--img_path', os.path.abspath(distorted), '--video', '--model-path', 'q-future/one-align'], work_dir=str(repo_path))

    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(distorted),
    }
    if result.returncode != 0:
        print(f"ERROR: QALIGN failed to run! {result.stderr}", force=True)
        return

    results.update(_parse_qalign_output(result.stdout))

    end_time = datetime.now()
    analysis_duration = end_time - start_time

    print_key_value("End Time", ts(end_time))
    print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
    print_key_value("Score", results.get("score", "N/A"), force=True)
    
    if output_file is not None:
        save_json(results, output_file)

    return results

def _parse_qalign_output(stdout):
    """Parse the output of the QALIGN model."""
    lines = stdout.strip().split("\n")
    results = {}
    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            try:
                value = float(line.strip('[]'))
                results['score'] = value
            except ValueError:
                continue
    return results