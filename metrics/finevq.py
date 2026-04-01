import os
import subprocess
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, print_separator, create_venv, run_in_venv, modify_file, is_quiet

def check_finevq(rebuild=False):
    """Check if FineVQ is available."""
    repo_path = Path(__file__).parent / "finevq"
    venv_path = repo_path / "venv"

    if rebuild and repo_path.exists():
        subprocess.run(['rm', '-rf', str(repo_path)], check=True)

    if not repo_path.exists():
        try:
            print_separator("BUILDING FineVQ", newline=True)
            print_line("Cloning FineVQ repository...")
            result = subprocess.run(['git', 'clone', 'https://github.com/IntMeGroup/FineVQ.git', repo_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

            print_line("Creating virtual environment...")
            create_venv(str(venv_path), python='python3.9', compile_decord=True, requirements=str(repo_path / 'requirements.txt'))
            #run_in_venv(str(venv_path), ['pip', 'install', 'flash-attn==2.3.6', '--no-build-isolation'])

            run_in_venv(str(venv_path), ['pip', 'install', 'wheel'])
            run_in_venv(str(venv_path), ['git', 'clone', 'https://github.com/Dao-AILab/flash-attention.git'], work_dir=str(repo_path))
            run_in_venv(str(venv_path), ['git', 'checkout', 'v2.3.6'], work_dir=str(repo_path / 'flash-attention'))
            run_in_venv(str(venv_path), ['python', 'setup.py', 'install'], work_dir=str(repo_path / 'flash-attention'))

            run_in_venv(str(venv_path), ['huggingface-cli', 'download', 'IntMeGroup/FineVQ_score', '--local-dir', str(repo_path / 'FineVQ_score')])

        except Exception as e:
            print_line(f"ERROR: Failed to build FineVQ - {e}", force=True)
            return False


    return True


def run_finevq(mode, distorted, output_dir=None):

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
    repo_path = Path(__file__).parent / "finevq"
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
        print_line(f"ERROR: FineVQ failed to run! {result.stderr}", force=True)
        return

    results.update(_parse_finevq_output(result.stdout))

    end_time = datetime.now()
    analysis_duration = end_time - start_time

    print_key_value("End Time", ts(end_time))
    print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
    print_key_value("Score", results.get("score", "N/A"))

    if is_quiet():
        print_line(f"FineVQ ({analysis_duration.total_seconds():.0f}s) | {results.get('score', 'N/A')} | {os.path.basename(distorted)}", force=True)
    
    if output_file is not None:
        save_json(results, output_file)

    return results

def _parse_finevq_output(stdout):
    """Parse the output of the FineVQ model."""
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