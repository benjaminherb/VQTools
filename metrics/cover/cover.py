import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, check_docker, build_docker_image


def check_cover():
    """Check if Docker and COVER image are available."""
    if not check_docker():
        print("ERROR: Docker is required for COVER but is not available")
        return False

    if not build_docker_image('cover:0.1.0', str(Path(__file__).parent)):
        print("ERROR: Failed to build COVER Docker image")
        return False
    
    return True


def run_cover(reference, distorted, mode, output_dir=None):
    """Run COVER video quality assessment."""
    
    # Prepare output file
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print(f"{output_file} exists already - SKIPPING!")
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        temp_fd, output_file = tempfile.mkstemp(suffix='.json', prefix='dover_')
        os.close(temp_fd)
    
    start_time = datetime.now()
    print("\nRESULTS")
    print_key_value("Start Time", ts(start_time))
    
    try:
        # Run COVER evaluation using evaluate_one_video
        distorted_name = os.path.basename(distorted)
        distorted_dir = os.path.dirname(distorted)
        absolute_distorted_dir = os.path.abspath(distorted_dir)
        
        # Check if file exists
        if not os.path.exists(distorted):
            print(f"ERROR: Video file does not exist: {distorted}")
            return None
            
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{absolute_distorted_dir}:/project/data',
            'cover:0.1.0',
            'python', '/project/COVER/evaluate_one_video.py',
            '-v', f'/project/data/{distorted_name}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: COVER evaluation failed: {result.stderr}")
            return None
        
        # Parse results from evaluate_one_video output
        results = _parse_results(result.stdout, result.stderr, distorted)
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        
        if results:
            print_key_value("Semantic Score", f"{results['semantic_score']:.4f}")
            print_key_value("Technical Score", f"{results['technical_score']:.4f}")
            print_key_value("Aesthetic Score", f"{results['aesthetic_score']:.4f}")
            print_key_value("Overall Score", f"{results['fused_score']:.4f}")
        
        if output_file.endswith('.json'):
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print(f"ERROR: COVER evaluation failed: {e}")
        return None


def _parse_results(stdout, stderr, video_path=None):
    """Parse COVER results from evaluate_one_video output."""
    # Format: path, semantic score, technical score, aesthetic score, overall/final score
    # Example: video_1.mp4,-0.086216,-0.089703,0.105127,-0.070792
    
    lines = stdout.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('path,') and ',' in line:
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    filename = parts[0]
                    semantic_score = float(parts[1])
                    technical_score = float(parts[2])
                    aesthetic_score = float(parts[3])
                    overall_score = float(parts[4])
                    
                    return {
                        'semantic_score': semantic_score,
                        'technical_score': technical_score,
                        'aesthetic_score': aesthetic_score,
                        'fused_score': overall_score,
                        'filename': os.path.basename(video_path) if video_path else filename,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                except (ValueError, IndexError):
                    continue
    
    return None