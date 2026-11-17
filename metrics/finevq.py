import csv
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, print_separator, create_venv, run_in_venv, is_quiet

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


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


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

    # FineVQ hard-codes model.cuda(), NCCL and flash-attn, so it cannot run on CPU/MPS.
    if str(get_device()) != 'cuda':
        print_line("ERROR: FineVQ requires a CUDA GPU - SKIPPING!", force=True)
        return None

    distorted = os.path.abspath(distorted)
    with tempfile.TemporaryDirectory(prefix='finevq_') as tmp:
        tmp = Path(tmp)
        video_list = tmp / 'videos.txt'
        video_list.write_text(os.path.basename(distorted) + '\n')
        meta_path = tmp / 'meta.json'
        meta_path.write_text(json.dumps({
            "vqtools": {
                "root": os.path.dirname(distorted),
                "video_name_txt": str(video_list),
                "data_augment": False,
                "repeat_time": 1,
                "length": 1,
            }
        }))
        csv_path = tmp / 'results.csv'

        # Same arguments as shell/infer.sh, with our own paths.
        command = [
            'torchrun', '--nnodes=1', '--node_rank=0', '--master_addr=127.0.0.1',
            '--nproc_per_node=1', f'--master_port={_free_port()}',
            'internvl/train/inference.py',
            '--model_name_or_path', str(WEIGHTS_PATH),
            '--conv_style', 'internlm2-chat',
            '--output_dir', str(tmp / 'work'),
            '--output_file', str(csv_path),
            '--metrics_file', str(tmp / 'metrics.txt'),
            '--meta_path', str(meta_path),
            '--overwrite_output_dir', 'True',
            '--force_image_size', '448',
            '--max_dynamic_patch', '6',
            '--down_sample_ratio', '0.5',
            '--drop_path_rate', '0.1',
            '--freeze_llm', 'True',
            '--freeze_mlp', 'True',
            '--freeze_backbone', 'True',
            '--vision_select_layer', '-1',
            '--dataloader_num_workers', '4',
            '--bf16', 'True',
            '--num_train_epochs', '100',
            '--per_device_train_batch_size', '1',
            '--gradient_accumulation_steps', '1',
            '--evaluation_strategy', 'no',
            '--save_strategy', 'steps',
            '--save_steps', '4800',
            '--save_total_limit', '1',
            '--learning_rate', '4e-5',
            '--weight_decay', '0.01',
            '--warmup_ratio', '0.03',
            '--lr_scheduler_type', 'cosine',
            '--logging_steps', '1',
            '--max_seq_length', '4096',
            '--do_train', 'True',
            '--grad_checkpoint', 'True',
            '--group_by_length', 'True',
            '--dynamic_image_size', 'True',
            '--use_thumbnail', 'True',
            '--ps_version', 'v2',
            '--deepspeed', 'zero_stage1_config.json',
            '--report_to', 'none',
        ]
        extra_env = {
            'PYTHONPATH': f"{REPO_PATH}:{os.environ.get('PYTHONPATH', '')}",
            'LAUNCHER': 'pytorch',
            'TF_CPP_MIN_LOG_LEVEL': '3',
        }
        result = run_in_venv(str(VENV_PATH), command, work_dir=str(REPO_PATH), extra_env=extra_env)

        if result is None or result.returncode != 0:
            stderr = result.stderr if result is not None else ''
            print_line(f"ERROR: FineVQ failed to run! {stderr}", force=True)
            return None

        score = _parse_finevq_csv(csv_path)
        if score is None:
            score = _parse_finevq_stdout(result.stdout)

    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(distorted),
    }
    if score is not None:
        results["score"] = score

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


def _parse_finevq_csv(csv_path):
    """Read the score from the CSV written by internvl/train/inference.py."""
    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                return float(row['pred_score'])
    except (OSError, KeyError, ValueError):
        pass
    return None


def _parse_finevq_stdout(stdout):
    """Fallback: the script prints 'pred_score <value>' for each video."""
    for line in stdout.splitlines():
        if line.startswith('pred_score'):
            try:
                return float(line.split()[-1])
            except ValueError:
                continue
    return None
