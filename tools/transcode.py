#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def convert_mkv_to_hevc(input_dir, output_dir, codec='ffvhuff', overwrite=False, dryrun=False):
    """Convert all MKV files to HEVC lossless"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mkv_files = list(input_path.glob('*.mkv'))
    mkv_files = [f for f in mkv_files if not f.name.startswith('.')]
    
    if not mkv_files:
        print(f"No MKV files found in '{input_path}'")
        return 0
    
    print(f"Found {len(mkv_files)} MKV files:")
    for mkv_file in mkv_files:
        print(f"  - {mkv_file.name}")
    
    success_count = 0
    
    for mkv_file in mkv_files:
        output_file = output_path / mkv_file.name

        if output_file.exists() and not overwrite:
            print(f"Skipping: {mkv_file.name} (already exists, use --overwrite to replace)")
            continue
        
        cmd = [
            'ffmpeg', '-i', str(mkv_file),
        ]
        
        if codec == 'h265':
            cmd.extend(['-c:v', 'libx265', '-x265-params', 'lossless=1'])
        elif codec == 'ffvhuff':
            cmd.extend(['-c:v', 'ffvhuff'])
        else:
            raise ValueError(f"Codec '{codec}' not implemented. Supported codecs: h265, ffvhuff")
            
        cmd.extend([
            '-c:a', 'copy', '-c:s', 'copy',
            '-y', str(output_file)
        ])
        
        print(f"\nConverting: {mkv_file.name}")
        print(f"Command: {' '.join(cmd)}")
        
        if not dryrun:
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Successfully converted {output_file}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error converting {mkv_file.name}:")
                print(f"Return code: {e.returncode}")
                print(e.stderr)
        else:
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Convert MKV files to HEVC lossless')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing MKV files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for converted files')
    parser.add_argument('--codec', choices=['ffvhuff', 'h265'], default='h265', 
                        help='Video codec to use (default: h265)')
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
    print(f"Overwrite: {args.overwrite}")
    print(f"Dry run: {args.dryrun}")
    
    success_count = convert_mkv_to_hevc(args.input, args.output, args.codec, args.overwrite, args.dryrun)
    
    total_files = len(list(input_path.glob('*.mkv')))
    print(f"\nConversion complete: {success_count}/{total_files} successful")
    
    return 0 if success_count == total_files else 1


if __name__ == "__main__":
    exit(main())
