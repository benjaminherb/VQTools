import os
import subprocess
import re
from datetime import datetime
from pathlib import Path
from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, modify_file, create_venv, run_in_venv, print_separator, transcode_video, is_quiet, get_device

def check_mdtvsfa(rebuild=False):
    """Clone the MDTVSFA repo if missing and download if needed. """

    repo = Path(__file__).parent / "mdtvsfa"

    if repo.exists() and rebuild:
        subprocess.run(['rm', '-rf', str(repo)], check=True)

    if not repo.exists():
        print_separator("BUILDING MDTVSFA", newline=True)
        print_line("Cloning MDTVSFA repository...", force=True)
        try:
            subprocess.run(['git', 'clone', 'https://github.com/lidq92/MDTVSFA.git', str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Modify test_demo.py to accept device argument
            modify_file(str(repo / 'test_demo.py'), [
                {'action': 'insert', 'line': 25, 'content': '    parser.add_argument("--device", "-d", type=str, default="cpu")'},
                {'action': 'replace', 'pattern': 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")', 'content': 'device = torch.device(args.device)'},
                {'action': 'replace', 'pattern': 'model.load_state_dict(torch.load(args.model_path))', 'content': 'model.load_state_dict(torch.load(args.model_path, map_location=device))'}
            ])
            
        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone MDTVSFA repository: {e}", force=True)
            return False

    if not (repo / 'venv').exists():
        print_line("Creating MDTVSFA virtual environment...")
        create_venv(str(repo / 'venv'), python='python3.8', compile_decord=False)
        run_in_venv(str(repo / 'venv'), ['pip', 'install', 'torch==2.4.1', 'torchvision==0.19.1', 'scikit-video==1.1.11', 'h5py==3.11.0', 'numpy==1.23.5'])

    return True


def run_mdtvsfa(mode, distorted, output_dir=None):
    """Run MDTVSFA video quality assessment."""
    
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
        
        repo = Path(__file__).parent / "mdtvsfa" 
        frame_batch_size = 1  # doesnt seem to affect performance much and avoids OOM with high res videos. Scores were identical in tests.
        device = get_device()
        cmd = [
            'python', str(repo / 'test_demo.py'),
            '--model_path', 'models/MDTVSFA.pt',
            '--device', str(device),
            '--video_path', distorted,
            '--frame_batch_size', str(frame_batch_size), # default
        ]
        result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(repo))

        if result.returncode != 0:
            print_line(f"ERROR: MDTVSFA evaluation failed: {result.stderr}", force=True)
            return None
        
        # Parse results from evaluate_one_video output
        score = _parse_mdtvfs_results(result.stdout, result.stderr, distorted)
        results = {
            'timestamp': ts(datetime.now()),
            'distorted': os.path.basename(distorted),
            'score': score
        }
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        
        if results:
            print_key_value(f"Score", f"{results['score']:.4f}")
            if is_quiet():
                print_line(f"Score ({analysis_duration.total_seconds():.0f}s) | {results['score']:.4f} | {os.path.basename(distorted)}", force=True)
        
        if output_file:
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: MDTVSFA evaluation failed: {e}", force=True)
        return None


def _parse_mdtvfs_results(stdout, stderr, video_path=None):
    """Parse MDTVSFA results from evaluate_one_video output."""
    
    # Predicted perceptual quality: [0.6550506]
    match = re.search(r'Predicted perceptual quality:\s*\[([0-9.]+)\]', stdout)
    if match:
        score = float(match.group(1))
        return score