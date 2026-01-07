
import os
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, print_separator, create_venv, run_in_venv, modify_file

def check_cvvdp(rebuild=False):
    """Check if ColorVideoVDP is available."""
    venv_path = Path(__file__).parent / "cvvdp_venv"

    if rebuild and venv_path.exists():
        subprocess.run(['rm', '-rf', str(venv_path)], check=True)

    if not venv_path.exists():
        try:
            print_line("Creating virtual environment...")
            create_venv(str(venv_path), python='python3.13')
            #run_in_venv(str(venv_path), ['conda', 'install', 'ffmpeg', 'conda-forge::freeimage'])
            if get_device() == 'cuda':
                run_in_venv(str(venv_path), ['conda', 'install', '-y', 'nvidia/label/cuda-12.9.1::cuda-toolkit'])
            run_in_venv(str(venv_path), ['pip', 'install', 'cvvdp'])

        except Exception as e:
            print_line(f"ERROR: Failed to build ColorVideoVDP - {e}", force=True)
            return False

    return True


def run_cvvdp(mode, distorted, reference, output_dir=None, display='standard_4k'):

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
    venv_path = Path(__file__).parent / "cvvdp_venv"
    #device = get_device()
    result = run_in_venv(str(venv_path), ['cvvdp', '--test', os.path.abspath(distorted), '--ref', os.path.abspath(reference), '--display', display, '-f', 'bicubic'])

    results = {
        "timestamp": ts(),
        "distorted": os.path.basename(distorted),
        "reference": os.path.basename(reference),
        "scaling": "bicubic",
    }

    if result.returncode != 0:
        print_line(f"ERROR: CVVDP failed to run! {result.stderr}", force=True)
        return

    output = result.stderr + result.stdout
    results.update(_parse_cvvdp_output(output))

    end_time = datetime.now()
    analysis_duration = end_time - start_time

    print_key_value("End Time", ts(end_time))
    print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
    print_key_value("Score", results.get("score", "N/A"), force=True)
    
    if output_file is not None:
        save_json(results, output_file)

    return results

import re


def _parse_cvvdp_output(stdout):
    """Parse CVVDP stdout and extract score and model metadata."""
    results = {}

    lines = stdout.strip().split("\n")
    for line in lines:
        line = line.strip()

        # [INFO] "ColorVideoVDP v0.5.4, 37.84 [pix/deg], Lpeak=200, Lblack=0.2, Lrefl=0.3979 [cd/m^2], (standard_fhd)"
        if "ColorVideoVDP" in line:
            # Remove INFO prefix and quotes
            info = line.split('"', 1)[-1].rsplit('"', 1)[0]

            m = re.search(r'(ColorVideoVDP v[0-9.]+)', info)
            if m:
                results['version'] = m.group(1)

            m = re.search(r'([\d.]+)\s*\[pix/deg\]', info)
            if m:
                results['pix_per_deg'] = float(m.group(1))

            m = re.search(r'Lpeak=([\d.]+)', info)
            if m:
                results['lpeak'] = float(m.group(1))

            m = re.search(r'Lblack=([\d.]+)', info)
            if m:
                results['lblack'] = float(m.group(1))

            m = re.search(r'\(([^)]+)\)\s*$', info)
            if m:
                results['display'] = m.group(1)

        # cvvdp=6.5130 [JOD]
        if line.startswith("cvvdp="):
            m = re.search(r'cvvdp=([\d.]+)', line)
            if m:
                results['score'] = float(m.group(1))

    return results