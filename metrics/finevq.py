import os
import subprocess
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, print_separator, create_venv, run_in_venv, is_quiet, \
    ModelWorker, WorkerError, get_worker, WORKERS_PATH

REPO_PATH = Path(__file__).parent / "finevq"
VENV_PATH = REPO_PATH / "venv"
WEIGHTS_PATH = REPO_PATH / "FineVQ_score"
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/"
                    "flash_attn-2.3.6+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl")


def check_finevq(rebuild=False):
    """Check if FineVQ is available."""
    if rebuild and REPO_PATH.exists():
        subprocess.run(['rm', '-rf', str(REPO_PATH)], check=True)

    if not REPO_PATH.exists():
        try:
            print_separator("BUILDING FineVQ", newline=True)
            print_line("Cloning FineVQ repository...")
            subprocess.run(['git', 'clone', 'https://github.com/IntMeGroup/FineVQ.git', str(REPO_PATH)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

            print_line("Creating virtual environment...")
            create_venv(str(VENV_PATH), python='python3.9', compile_decord=True)

            venv = str(VENV_PATH)
            run_in_venv(venv, ['pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2', 'numpy<2', 'importlib_metadata'])
            run_in_venv(venv, ['pip', 'install', '-r', str(REPO_PATH / 'requirements.txt')])
            # torch 2.1 needs pkg_resources, which newer setuptools no longer ship
            run_in_venv(venv, ['pip', 'install', 'setuptools<70'])
            run_in_venv(venv, ['pip', 'install', FLASH_ATTN_WHEEL])

            run_in_venv(venv, ['python', '-c',
                               'import sys; from huggingface_hub import snapshot_download; '
                               'snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])',
                               'IntMeGroup/FineVQ_score', str(WEIGHTS_PATH)])
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

    # FineVQ hard-codes model.cuda() and flash-attn, so it cannot run on CPU/MPS.
    if str(get_device()) != 'cuda':
        print_line("ERROR: FineVQ requires a CUDA GPU - SKIPPING!", force=True)
        return None

    try:
        # the model is loaded once and reused for all videos of this vqcheck run
        worker = get_worker(('finevq',), lambda: ModelWorker(
            'FineVQ', VENV_PATH, WORKERS_PATH / 'finevq_worker.py', [WEIGHTS_PATH], work_dir=REPO_PATH,
            extra_env={'PYTHONPATH': f"{REPO_PATH}:{os.environ.get('PYTHONPATH', '')}", 'TF_CPP_MIN_LOG_LEVEL': '3'}))
    except WorkerError as e:
        print_line(f"ERROR: FineVQ failed to load! {e}", force=True)
        return None

    start_time = datetime.now()

    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))

    distorted = os.path.abspath(distorted)
    try:
        score = worker.request(video=distorted)['score']
    except WorkerError as e:
        print_line(f"ERROR: FineVQ failed to run! {e}", force=True)
        return None

    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(distorted),
        "score": score,
    }

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
