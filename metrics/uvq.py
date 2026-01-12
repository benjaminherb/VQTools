import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import re
import json

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_video_info, print_line, print_separator, create_venv, run_in_venv, transcode_video


def check_uvq(rebuild=False):
    """Check if UVQ repository exists and is properly set up."""
    repo_path = Path(__file__).parent / "uvq"
    venv_path = repo_path / "venv"

    if rebuild and repo_path.exists():
        subprocess.run(['rm', '-rf', str(repo_path)], check=True)
    
    if not repo_path.exists():
        print_separator("BUILDING UVQ", newline=True)
        print_line("Cloning UVQ repository...")
        try:
            result = subprocess.run(['git', 'clone', 'https://github.com/google/uvq.git', str(repo_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print_line(f"ERROR: Failed to clone UVQ repository: {result.stderr}", force=True)
                return False
            print_line("Setting up UVQ virtual environment...")
            create_venv(str(venv_path), 'python3.12', requirements=str(repo_path / "requirements.txt")) 

        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone UVQ repository: {e}", force=True)
            return False
    return True

def run_uvq_command(distorted, mode):
    uvq_work_dir = Path(__file__).parent / "uvq"
    model_version = '1.0' if mode == 'uvq' else '1.5'
    cmd = [
        'python', str(uvq_work_dir / "uvq_inference.py"),
        f'{os.path.abspath(distorted)}',
        f'--model_version', model_version,
        f'--output_all_stats',
    ]

    result = run_in_venv(str(uvq_work_dir / 'venv'), cmd, work_dir=str(uvq_work_dir))
    
    return result


def run_uvq(mode, distorted, output_dir=None):
    """Run UVQ video quality assessment."""
    
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
            result = run_uvq_command(distorted, mode)

            if result.returncode != 0:
                print_key_value("Transcoding Input", "True")
                transcoded_path = Path(temp_dir) / "distorted.mkv"
                transcode_video(distorted, transcoded_path)
                result = run_uvq_command(str(transcoded_path), mode)

            if result.returncode != 0:
                print_line(f"ERROR: UVQ evaluation failed: {result.stderr}", force=True)
                return None

            results = {
                "timestamp": ts(),
                "distorted": os.path.basename(distorted),
            }
            results.update(_parse_uvq_results(result.stdout))
            
            end_time = datetime.now()
            analysis_duration = end_time - start_time
            
            print_key_value("End Time", ts(end_time))
            print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
            if results and mode == 'uvq':
                print_key_value("UVQ Score", f"{results['compression_content_distortion']:.4f}", force=True)
                print_key_value("Compression", f"{results['compression']:.4f}")
                print_key_value("Content", f"{results['content']:.4f}")
                print_key_value("Distortion", f"{results['distortion']:.4f}")
                print_key_value("Compression+Content", f"{results['compression_content']:.4f}")
                print_key_value("Compression+Distortion", f"{results['compression_distortion']:.4f}")
                print_key_value("Content+Distortion", f"{results['content_distortion']:.4f}")
                print_key_value("Compression+Content+Distortion", f"{results['compression_content_distortion']:.4f}")
            elif results and mode == 'uvq1p5':
                print_key_value("UVQ 1.5 Score", f"{results['uvq1p5_score']:.4f}", force=True)

            if output_file:
                save_json(results, output_file)
            
            return results
            
    except Exception as e:
        print_line(f"ERROR: UVQ evaluation failed: {e}", force=True)
        return None

def _parse_uvq_results(stdout):
    match = re.search(r'\{.*?\}', stdout, re.DOTALL)
    if not match:
        raise ValueError(f"Failed to extract UVQ results from output ({stdout})")

    dict_str = match.group(0).strip()

    try:
        results = json.loads(dict_str)
        # delet if in there 'video_name
        if 'video_name' in results:
            del results['video_name'] # avoid redundant info

        return results
    except json.JSONDecodeError:
        raise ValueError(f"Failed to extract UVQ results from output ({stdout})")