import os
import subprocess
import tempfile
import csv
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, check_docker, build_docker_image, print_line


def check_maxvqa():
    """Check if Docker and MAXVQA image are available."""
    if not check_docker():
        print_line("ERROR: Docker is required for MAXVQA but is not available", force=True)
        return False

    if not build_docker_image('maxvqa:0.1.0', str(Path(__file__).parent)):
        print_line("ERROR: Failed to build MAXVQA Docker image", force=True)
        return False

    return True


def run_maxvqa(mode, distorted, reference, output_dir=None):
    """Run MAXVQA video quality assessment."""

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
        distorted_name = os.path.basename(distorted)
        distorted_dir = os.path.dirname(distorted)
        absolute_distorted_dir = os.path.abspath(distorted_dir)

        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{absolute_distorted_dir}:/project/data',
            'maxvqa:0.1.0',
            'python', 'demo_maxvqa.py',
            '/project/data/' + distorted_name
        ]
        print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print_line(f"ERROR: MAXVQA evaluation failed: {result.stderr}", force=True)
            return None

        # Parse results from demo_maxvqa output
        results = _parse_maxvqa_results(result.stdout, result.stderr, distorted)

        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")

        if results:
            print_key_value("Overall Score", f"{results['overall_score']:.4f}", force=True)
            if 'dimensions' in results:
                for dim, score in results['dimensions'].items():
                    print_key_value(f"{dim.title()} Score", f"{score:.4f}")

        if output_file:
            save_json(results, output_file)

        return results

    except Exception as e:
        print_line(f"ERROR: MAXVQA evaluation failed: {e}", force=True)
        return None


def _parse_maxvqa_results(stdout, stderr, video_path=None):
    """Parse MAXVQA results from demo_maxvqa output."""
    print("STDOUT:", stdout[:500])
    print("STDERR:", stderr[:500])
    return None
