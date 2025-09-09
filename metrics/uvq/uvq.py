import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_video_info, print_line


def check_uvq():
    """Check if UVQ repository exists and is properly set up."""
    repo_path = Path(__file__).parent / "uvq"
    if not repo_path.exists():
        print_line("Cloning UVQ repository...", force=True)
        try:
            result = subprocess.run(['git', 'clone', 'https://github.com/google/uvq.git', str(repo_path)], check=True)
            if result.returncode != 0:
                print_line(f"ERROR: Failed to clone UVQ repository: {result.stderr}", force=True)
                return False
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone UVQ repository: {e}", force=True)
            return False
    return True


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
        video_id = os.path.splitext(os.path.basename(distorted))[0]
        video_length = int(round(get_video_info(distorted)['duration']))

        with tempfile.TemporaryDirectory() as temp_dir:
            uvq_output_dir = os.path.join(temp_dir, 'uvq_results')
            uvq_work_dir = Path(__file__).parent / "uvq"
            
            cmd = [
                'python3', str(uvq_work_dir / "uvq_main.py"),
                f'--input_files={video_id},{video_length},{os.path.abspath(distorted)}',
                f'--output_dir={uvq_output_dir}',
                f'--model_dir={str(uvq_work_dir / "models")}',
                '--transpose=False'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=uvq_work_dir)
            
            if result.returncode != 0:
                print_line(f"ERROR: UVQ evaluation failed: {result.stderr}", force=True)
                return None
            
            results = _parse_uvq_results(uvq_output_dir, distorted)
            
            end_time = datetime.now()
            analysis_duration = end_time - start_time
            
            print_key_value("End Time", ts(end_time))
            print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
            if results:
                print_key_value("UVQ Score", f"{results['compression_content_distortion']:.4f}", force=True)
                print_key_value("Compression", f"{results['compression']:.4f}")
                print_key_value("Content", f"{results['content']:.4f}")
                print_key_value("Distortion", f"{results['distortion']:.4f}")
                print_key_value("Compression+Content", f"{results['compression_content']:.4f}")
                print_key_value("Compression+Distortion", f"{results['compression_distortion']:.4f}")
                print_key_value("Content+Distortion", f"{results['content_distortion']:.4f}")
                print_key_value("Compression+Content+Distortion", f"{results['compression_content_distortion']:.4f}")

            if output_file:
                save_json(results, output_file)
            
            return results
            
    except Exception as e:
        print_line(f"ERROR: UVQ evaluation failed: {e}", force=True)
        return None

def _parse_uvq_results(output_dir, video_path):
    """Parse UVQ results from output directory."""
    import csv
    
    try:
        # results/video_id/video_id_uvq.csv
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        csv_file = os.path.join(output_dir, video_id, f"{video_id}_uvq.csv")
        
        scores = {
            'timestamp': ts(),
            'distorted': os.path.basename(video_path)
        }
        
        with open(csv_file, 'r') as f:
            for row in csv.reader(f):
                model_name = row[1]  # compression, content, distortion, etc.
                score = float(row[2])
                scores[model_name] = score

        scores['uvq'] = scores.get('compression_content_distortion', None)
        return scores    
    except Exception as e:
        print_line(f"ERROR: Could not parse UVQ results: {e}", force=True)

    return None