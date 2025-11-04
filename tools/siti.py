#!/usr/bin/env python3
import argparse
import subprocess
import shutil
from pathlib import Path
import multiprocessing

VIDEO_EXTENSIONS = ['.mkv', '.mp4', '.mov', '.avi', '.m4v', '.wmv', '.yuv', '.y4m', '.raw']


def build_siti():
    if not shutil.which('uvx'):
        raise EnvironmentError("uvx command not found")

    cmd = ['uvx', 'siti-tools']
    return subprocess.run(cmd, check=True)


def run_siti(input_file, output_file, overwrite=False, color_range='limited'):
    if output_file.exists() and not overwrite:
        print(f"Skipping: {input_file.name}")
        return False

    cmd = ['uvx', 'siti-tools', str(input_file), "-o", str(output_file), '-f', 'json', '--color-range', color_range]
    print(f"Analyzing {input_file.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_lines = (result.stderr or result.stdout or '').strip().splitlines()
        err = err_lines[-1] if err_lines else 'unknown error'
        print(f"Error processing {input_file.name}: {err}")
        return False
    return True


def run_task_worker(task):
    input_path_str, output_path_str, overwrite_flag, crange = task
    try:
        success = run_siti(Path(input_path_str), Path(output_path_str), overwrite=overwrite_flag, color_range=crange)
        return (input_path_str, success, None)
    except Exception as e:
        return (input_path_str, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Convert video files to standardized format (lossless)')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing video files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for converted files')
    parser.add_argument('--color-range', default='limited', choices=['full', 'limited'], help='Color range for processing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes (0 = serial)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix in VIDEO_EXTENSIONS:
        video_files = [input_path]
    else:
        video_files = list(input_path.glob('*'))
        video_files = [f for f in video_files if f.suffix in VIDEO_EXTENSIONS and not f.name.startswith('.')]

    print(f"Processing {len(video_files)} video files")

    tasks = []
    for video_file in video_files:
        output_file = output_path / f"{video_file.stem}.siti.json"
        tasks.append((str(video_file), str(output_file), args.overwrite, args.color_range))


    if args.workers and args.workers > 0:
        with multiprocessing.Pool(args.workers) as pool:
            for input_path_str, success, error_msg in pool.imap_unordered(run_task_worker, tasks):
                if not success:
                    print(f"Error processing {Path(input_path_str).name}: {error_msg}")
    else:
        for task in tasks:
            input_path_str, success, error_msg = run_task_worker(task)
            if not success:
                print(f"Error processing {Path(input_path_str).name}: {error_msg}")


if __name__ == "__main__":
    main()
