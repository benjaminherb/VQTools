import os
import subprocess
import tempfile
import re
from datetime import datetime
from pathlib import Path
import urllib

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, modify_file, create_venv, run_in_venv, print_separator, transcode_video

MODEL_FILES = [
    ("DOVER.pth", "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"),
    ("DOVER-Mobile.pth", "https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth"),
]

def check_dover():
    """Clone the Dover repo if missing and download pretrained weights if needed. """

    repo = Path(__file__).parent / "dover"
    if not repo.exists():
        print_separator("BUILDING DOVER", newline=True)
        print_line("Cloning Dover repository...", force=True)
        try:
            subprocess.run(['git', 'clone', 'https://github.com/QualityAssessment/DOVER', str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Modify dover_datasets.py to support more file extensions
            datasets_file = repo / 'dover' / 'datasets' / 'dover_datasets.py'
            modify_file(str(datasets_file), [{'action': 'replace', 'pattern': 'if file.endswith(".mp4"):', 'content': 'if file.endswith((".mp4", ".mkv", ".mov")):'}])
            
            # Set num_workers to 0 in dover.yml
            yml_file = repo / 'dover.yml'
            modify_file(str(yml_file), [{'action': 'replace', 'pattern': 'num_workers:', 'content': 'num_workers: 0'}])
            
            # Add random seed imports and initialization to evaluate_one_video.py
            eval_file = repo / 'evaluate_one_video.py'
            modify_file(str(eval_file), [
                {'action': 'insert', 'pattern': 'import torch', 'content': [
                    'import random',
                    'import numpy as np',
                    'torch.manual_seed(42)',
                    'np.random.seed(42)',
                    'random.seed(42)'
                ]      },
                    {'action': 'delete', 'from': 137, 'to': -1},
                    {'action': 'insert', 'line': 136, 'content': [
                        '    import json',
                        '    output_data = {',
                        '        "technical_score": results[0],',
                        '        "aesthetic_score": results[1],',
                        '        "fused_score": fuse_results(results)',
                        '    }',
                        '    print("DOVER_RESULTS_JSON:" + json.dumps(output_data))'
                    ]}
                ])
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone Dover repository: {e}", force=True)
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

    except Exception as e:
        print_line(f"ERROR: Failed to download pretrained models: {e}", force=True)
        return False
    
    if not (repo / 'venv').exists():
        print_line("Creating DOVER virtual environment...")
        create_venv(str(repo / 'venv'), python='python3.8', requirements=str(repo / 'requirements.txt'), compile_decord=True)
        run_in_venv(str(repo / 'venv'), ['pip', 'install', '-e', str(repo)])

    return True


def run_dover(mode, distorted, output_dir=None):
    """Run DOVER video quality assessment."""
    
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
    distorted = os.path.abspath(distorted)
    
    try:
        repo = Path(__file__).parent / "dover" 
        cmd = [
            'python', str(repo / 'evaluate_one_video.py'),
            '-d', 'cpu',
            '-v', f'{distorted}',
            '-f'
        ]
        result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(repo))
        
        if result.returncode != 0 and 'DECORDError' in result.stderr:
            print_line("Transcoding input video to a compatible format...", force=True)
            with tempfile.TemporaryDirectory() as temp_dir:
                transcoded_path = Path(temp_dir) / "distorted.mkv"
                transcode_video(distorted, transcoded_path)
                cmd[4] = os.path.abspath(transcoded_path)
                result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(repo))
        

        if result.returncode != 0:
            print_line(f"ERROR: DOVER evaluation failed: {result.stderr}", force=True)
            return None
        
        # Parse results from evaluate_one_video output
        results = _parse_dover_results_single(result.stdout, result.stderr, distorted)
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        
        if results:
            print_key_value("Technical Score", f"{results['technical_score']:.4f}")
            print_key_value("Aesthetic Score", f"{results['aesthetic_score']:.4f}")
            print_key_value("Fused Score", f"{results['fused_score']:.4f}", force=True)
        
        if output_file:
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: DOVER evaluation failed: {e}", force=True)
        return None


def _parse_dover_results_single(stdout, stderr, video_path=None):
    """Parse DOVER results from evaluate_one_video output."""
    import re
    import json
    
    # Look for structured JSON output first
    json_match = re.search(r'DOVER_RESULTS_JSON:(\{.*\})', stdout)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return {
                'timestamp': ts(),
                'distorted': os.path.basename(video_path) if video_path else '',
                'technical_score': data.get('technical_score'),
                'aesthetic_score': data.get('aesthetic_score'),
                'fused_score': data.get('fused_score')
            }
        except (json.JSONDecodeError, KeyError):
            pass