#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
import multiprocessing

VIDEO_EXTENSIONS = ['.mkv', '.mp4', '.mov', '.avi', '.m4v', '.wmv', '.yuv', '.y4m', '.raw']

VCA_REPO = 'https://github.com/cd-athena/VCA.git'
# Built next to tools/ (repo root locally, ~/.local/share/vqtools when installed)
VCA_DIR = Path(__file__).resolve().parent.parent / 'vca'
VCA_BIN = VCA_DIR / 'build' / 'source' / 'apps' / 'vca' / 'vca'

REQUIREMENTS = """VCA build requirements:
  - git
  - CMake >= 3.13
  - C++11 compiler (GCC >= 4.8, Clang >= 3.3)
  - make (or another CMake generator)
  - NASM (x86 SIMD support)"""


def build_vca(rebuild=False):
    if rebuild and VCA_DIR.exists():
        print(f"Removing existing VCA build at {VCA_DIR}")
        shutil.rmtree(VCA_DIR)

    if VCA_BIN.exists():
        return VCA_BIN

    missing = [c for c in ['git', 'cmake', 'nasm'] if not shutil.which(c)]
    if not any(shutil.which(c) for c in ['c++', 'g++', 'clang++']):
        missing.append('C++ compiler')
    if missing:
        print(f"ERROR: Missing build tools: {', '.join(missing)}")
        print(REQUIREMENTS)
        sys.exit(1)

    print(f"Building VCA in {VCA_DIR}")
    build_dir = VCA_DIR / 'build'
    steps = []
    if not VCA_DIR.exists():
        steps.append(['git', 'clone', VCA_REPO, str(VCA_DIR)])
    steps.append(['cmake', '-S', str(VCA_DIR), '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release'])
    steps.append(['cmake', '--build', str(build_dir), '-j', '--target', 'vca'])

    # A copied build dir has a stale CMake cache, so always configure from scratch
    shutil.rmtree(build_dir, ignore_errors=True)
    for cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print((result.stderr or result.stdout).strip())
            print(f"ERROR: VCA build step failed: {' '.join(cmd)}")
            print(REQUIREMENTS)
            sys.exit(1)

    if not VCA_BIN.exists():
        print(f"ERROR: VCA binary not found after build ({VCA_BIN})")
        sys.exit(1)
    print("VCA build complete")
    return VCA_BIN


def read_csv(path):
    with open(path, newline='') as f:
        rows = [[c.strip() for c in row] for row in csv.reader(f) if row]
    if not rows:
        return [], []
    return rows[0], [[float(v) for v in row] for row in rows[1:]]


def run_vca(input_file, output_file, overwrite=False, pix_fmt='yuv420p', threads=0, block_size=32):
    if output_file.exists() and not overwrite:
        print(f"Skipping: {input_file.name}")
        return False

    print(f"Analyzing {input_file.name}")
    with tempfile.TemporaryDirectory(prefix='vca_') as tmp:
        complexity_csv = Path(tmp) / 'complexity.csv'
        shot_csv = Path(tmp) / 'shots.csv'

        ffmpeg_cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(input_file),
                      '-map', '0:v:0', '-pix_fmt', pix_fmt, '-f', 'yuv4mpegpipe', '-strict', '-1', '-']
        vca_cmd = [str(VCA_BIN), '--input', 'stdin', '--y4m',
                   '--complexity-csv', str(complexity_csv), '--shot-csv', str(shot_csv),
                   '--threads', str(threads), '--block-size', str(block_size)]

        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.run(vca_cmd, stdin=ffmpeg.stdout, capture_output=True, text=True)
        ffmpeg.stdout.close()
        ffmpeg_err = ffmpeg.stderr.read().decode(errors='replace')
        ffmpeg.wait()

        if ffmpeg.returncode != 0 or result.returncode != 0 or not complexity_csv.exists():
            err_lines = (ffmpeg_err if ffmpeg.returncode != 0 else (result.stderr or result.stdout or '')).strip().splitlines()
            err = err_lines[-1] if err_lines else 'unknown error'
            print(f"Error processing {input_file.name}: {err}")
            return False

        header, rows = read_csv(complexity_csv)
        frames = {name: [row[i] for row in rows] for i, name in enumerate(header)}
        frames['POC'] = [int(v) for v in frames['POC']]

        shots = []
        if shot_csv.exists():
            shot_header, shot_rows = read_csv(shot_csv)
            shots = [dict(zip(shot_header, row)) for row in shot_rows]
            for shot in shots:
                shot['ID'] = int(shot['ID'])
                shot['Start POC'] = int(shot['Start POC'])

    version = re.search(r'Video Complexity Analyzer (\S+)', result.stdout + result.stderr)
    output = {
        'file': input_file.name,
        'vca_version': version.group(1) if version else None,
        'settings': {'pix_fmt': pix_fmt, 'block_size': block_size},
        'num_frames': len(rows),
        'mean': {k: sum(v) / len(v) for k, v in frames.items() if k != 'POC' and v},
        'frames': frames,
        'shots': shots,
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    return True


def run_task_worker(task):
    input_path_str, output_path_str, overwrite_flag, pix_fmt, threads, block_size = task
    try:
        success = run_vca(Path(input_path_str), Path(output_path_str), overwrite=overwrite_flag,
                          pix_fmt=pix_fmt, threads=threads, block_size=block_size)
        return (input_path_str, success, None)
    except Exception as e:
        return (input_path_str, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Calculate VCA (Video Complexity Analyzer) features for video files')
    parser.add_argument('-i', '--input', help='Input video file or directory containing video files')
    parser.add_argument('-o', '--output', help='Output directory for *.vca.json files')
    parser.add_argument('--pix-fmt', default='yuv420p',
                        help='Pixel format passed to VCA (default: yuv420p, e.g. yuv420p10le, yuv444p)')
    parser.add_argument('--block-size', type=int, default=32, choices=[8, 16, 32], help='DCT block size')
    parser.add_argument('--threads', type=int, default=0, help='VCA threads per video (0 = autodetect)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes (0 = serial)')
    parser.add_argument('--rebuild', action='store_true', help='Delete and rebuild VCA')
    parser.add_argument('--build-only', action='store_true', help='Only build VCA and exit')

    args = parser.parse_args()

    build_vca(rebuild=args.rebuild)
    if args.build_only:
        return
    if not args.input or not args.output:
        parser.error('-i/--input and -o/--output are required')
    if not shutil.which('ffmpeg'):
        raise EnvironmentError("ffmpeg command not found")

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
        output_file = output_path / f"{video_file.stem}.vca.json"
        tasks.append((str(video_file), str(output_file), args.overwrite, args.pix_fmt, args.threads, args.block_size))

    if args.workers and args.workers > 0:
        with multiprocessing.Pool(args.workers) as pool:
            for input_path_str, success, error_msg in pool.imap_unordered(run_task_worker, tasks):
                if not success and error_msg:
                    print(f"Error processing {Path(input_path_str).name}: {error_msg}")
    else:
        for task in tasks:
            input_path_str, success, error_msg = run_task_worker(task)
            if not success and error_msg:
                print(f"Error processing {Path(input_path_str).name}: {error_msg}")


if __name__ == "__main__":
    main()
