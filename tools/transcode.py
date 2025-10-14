#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = ['.mkv', '.mp4', '.mov', '.avi', '.m4v', '.wmv']

def transcode(input_dir, output_dir, codec='ffvhuff', scale=None, overwrite=False, dryrun=False):
    """Convert all video files to a specified codec"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = list(input_path.glob('*'))
    video_files = [f for f in video_files if f.suffix in VIDEO_EXTENSIONS and not f.name.startswith('.')]

    if not video_files:
        print(f"No video files found in '{input_path}'")
        return 0

    print(f"Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    success_count = 0

    for video_file in video_files:
        output_file = output_path / f"{video_file.stem}.mkv"

        if output_file.exists() and not overwrite:
            print(f"Skipping: {video_file.name} (already exists, use --overwrite to replace)")
            continue
        
        cmd = [
            'ffmpeg', '-i', str(video_file),
        ]
        
        if codec == 'h265':
            cmd.extend(['-c:v', 'libx265', '-x265-params', 'lossless=1'])
        elif codec == 'ffvhuff':
            cmd.extend(['-c:v', 'ffvhuff'])
        else:
            raise ValueError(f"Codec '{codec}' not implemented. Supported codecs: h265, ffvhuff")

        if scale is not None:
            width, height = scale
            cmd.extend(['-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease', '-sws_flags', 'bicubic'])
            
        cmd.extend([
            '-c:a', 'copy', '-c:s', 'copy',
            '-y', str(output_file)
        ])
        
        print(f"\nConverting: {video_file.name}")
        print(f"Command: {' '.join(cmd)}")
        
        if not dryrun:
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Successfully converted {output_file}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error converting {video_file.name}:")
                print(f"Return code: {e.returncode}")
                print(e.stderr)
        else:
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Convert video files to standardized format (lossless)')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing video files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for converted files')
    parser.add_argument('--codec', choices=['ffvhuff', 'h265'], default='h265', 
                        help='Video codec to use (default: h265)')
    parser.add_argument('--scale', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Scale videos to specified WIDTH and HEIGHT (e.g., --scale 1920 1080)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--dryrun', action='store_true', help='Show commands without executing them')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' is not a directory or does not exist.")
        return 1
    
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Codec: {args.codec}")
    if args.scale:
        print(f"Scale: {args.scale[0]}x{args.scale[1]}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Dry run: {args.dryrun}")
    
    success_count = transcode(args.input, args.output, args.codec, args.scale, args.overwrite, args.dryrun)

    total_files = len([f for f in list(input_path.glob('*')) if f.suffix in VIDEO_EXTENSIONS and not f.name.startswith('.')])
    print(f"\nConversion complete: {success_count}/{total_files} successful")
    
    return 0 if success_count == total_files else 1


if __name__ == "__main__":
    exit(main())
