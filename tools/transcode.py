#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = ['.mkv', '.mp4', '.mov', '.avi', '.m4v', '.wmv', '.yuv', '.y4m', '.raw']

def transcode(args):
    """Convert all video files to a specified codec"""
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    input_is_raw = all([args.input_resolution, args.input_framerate, args.input_pixel_format])

    if input_path.is_file() and input_path.suffix in VIDEO_EXTENSIONS:
        video_files = [input_path]
    else:
        video_files = list(input_path.glob('*'))
        video_files = [f for f in video_files if f.suffix in VIDEO_EXTENSIONS and not f.name.startswith('.')]

    if not video_files:
        print(f"No video files found in '{input_path}'")
        return 0

    print(f"Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    success_count = 0

    for i, video_file in enumerate(video_files):
        output_file = output_path / f"{video_file.stem}.mkv"
        print(f"\nTask {i+1}/{len(video_files)}")

        if output_file.exists() and not args.overwrite:
            print(f"Skipping:   {video_file.name} (already exists, use --overwrite to replace)")
            continue
        
        cmd = ['ffmpeg', '-i', str(video_file)]
        if input_is_raw:
            cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-pixel_format', args.input_pixel_format,
                '-video_size', f"{args.input_resolution[0]}x{args.input_resolution[1]}",
                '-framerate', str(args.input_framerate),
                '-i', str(video_file)
            ]
        
        if args.codec == 'h265':
            cmd.extend(['-c:v', 'libx265', '-x265-params', 'lossless=1'])
        elif args.codec == 'ffvhuff':
            cmd.extend(['-c:v', 'ffvhuff'])
        elif args.codec == 'ffv1':
            cmd.extend(['-c:v', 'ffv1', '-level', '3', '-slicecrc', '1'])
        elif args.codec == 'preview':
            cmd.extend(['-c:v', 'libx264', '-crf', '26', '-preset', 'fast'])
        else:
            raise ValueError(f"Codec '{args.codec}' not implemented. Supported codecs: h265, ffvhuff, ffv1")

        if args.scale is not None:
            width, height = args.scale
            cmd.extend(['-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease', '-sws_flags', 'bicubic'])
            
        cmd.extend([
            '-c:a', 'copy', '-c:s', 'copy',
            '-y', str(output_file)
        ])
        
        print(f"Converting: {video_file.name}")
        print(f"Command:    {' '.join(cmd)}")
        
        if not args.dryrun:
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
    parser.add_argument('-c', '--codec', choices=['ffvhuff', 'h265', 'ffv1', 'preview'], default='ffv1', 
                        help='Video codec to use (default: ffv1)')
    parser.add_argument('--scale', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Scale videos to specified WIDTH and HEIGHT (e.g., --scale 1920 1080)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--input_resolution', '-ir', nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Input resolution for raw video files')
    parser.add_argument('--input_framerate', '-ifr', type=float, help='Input framerate for raw video files')
    parser.add_argument('--input_pixel_format', '-ipf', type=str, help='Input pixel format for raw video files (e.g., yuv420p)')
    parser.add_argument('--dryrun', action='store_true', help='Show commands without executing them')
    
    args = parser.parse_args()

    if any([args.input_resolution, args.input_framerate, args.input_pixel_format]) and not all([args.input_resolution, args.input_framerate, args.input_pixel_format]):
        parser.error("When specifying input format options (for raw input), all of --input_resolution, --input_framerate, and --input_pixel_format must be provided.")
    
    input_path = Path(args.input)
    if not input_path.exists() or (not input_path.is_dir() and not input_path.is_file()):
        print(f"Error: Input '{input_path}' does not exist.")
        return 1
    
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Codec:     {args.codec}")
    if args.scale:
        print(f"Scale:     {args.scale[0]}x{args.scale[1]}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Dry run:   {args.dryrun}")
    print()
    
    success_count = transcode(args)

    total_files = 1
    if input_path.is_dir():
        total_files = len([f for f in list(input_path.glob('*')) if f.suffix in VIDEO_EXTENSIONS and not f.name.startswith('.')])
    print(f"\nConversion complete: {success_count}/{total_files} successful")
    
    return 0 if success_count == total_files else 1


if __name__ == "__main__":
    exit(main())
