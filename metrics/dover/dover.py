import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, check_docker, build_docker_image, print_line


def check_dover():
    """Check if Docker and DOVER image are available."""
    if not check_docker():
        print_line("ERROR: Docker is required for DOVER but is not available", force=True)
        return False

    if not build_docker_image('dover:0.1.0', str(Path(__file__).parent)):
        print_line("ERROR: Failed to build DOVER Docker image", force=True)
        return False
    
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
    
    try:
        # Run DOVER evaluation using evaluate_one_video
        distorted_name = os.path.basename(distorted)
        # Debug: list files in the directory
        distorted_dir = os.path.dirname(distorted)
        absolute_distorted_dir = os.path.abspath(distorted_dir)
        
        # Check if file exists
        if not os.path.exists(distorted):
            print_line(f"ERROR: Video file does not exist: {distorted}", force=True)
            return None
            
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{absolute_distorted_dir}:/project/data',
            'dover:0.1.0',
            'python', '/project/DOVER/evaluate_one_video.py',
            '-d', 'cpu',
            '-v', f'/project/data/{distorted_name}',
            '-f'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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