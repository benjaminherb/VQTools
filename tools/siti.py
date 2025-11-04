#!/usr/bin/env python3
import argparse
import subprocess
import shutil
from pathlib import Path

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
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error processing {input_file.name}: {result.stderr}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert video files to standardized format (lossless)')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing video files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for converted files')
    parser.add_argument('--color-range', default='limited', choices=['full', 'limited'], help='Color range for processing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
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
    for video_file in video_files:
        output_file = output_path / f"{video_file.stem}.siti.json"
        try:
            run_siti(video_file, output_file, overwrite=args.overwrite, color_range=args.color_range)
        except Exception as e:
            print(f"Error processing {video_file.name}: {repr(e)}")

if __name__ == "__main__":
    main()
