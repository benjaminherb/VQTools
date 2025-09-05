#!/usr/bin/env python3
import sys
import subprocess
import os
import argparse
import cv2
import json
from metrics import run_lpips, run_ffmpeg, run_cvqa
from metrics.utils import get_video_files, find_reference_file, format_duration, format_file_size, print_separator, print_key_value

MODES = {
    'ffmpeg': ['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr'],
    'cvqa': ['cvqa-nr', 'cvqa-nr-ms', 'cvqa-fr', 'cvqa-fr-ms'],
    'lpips': ['lpips'],
    'check': ['check']
}
FR_MODES = ['check', 'vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr', 'check', 'cvqa-fr', 'cvqa-fr-ms', 'lpips']
NR_MODES = ['cvqa-nr', 'cvqa-nr-ms']


def get_video_info(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        '-select_streams', 'v:0',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data['streams']:
            raise Exception(f"No video stream found in {video_path}")
            
        stream = data['streams'][0]
        format_info = data.get('format', {})
        fps_str = stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
        frame_count = get_frame_count_cv2(video_path)
        duration = frame_count / fps if fps > 0 else 0
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))

        return {
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'pix_fmt': stream.get('pix_fmt', 'unknown'),
            'color_range': stream.get('color_range', 'unknown'),
            'file_size': int(format_info.get('size', 0)),
            'frame_count': frame_count,
            'duration': duration
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe on {video_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output for {video_path}: {e}")
        return None


def get_frame_count_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count



def compare_video_properties(reference, distorted, verbose=True):
    ref_info = get_video_info(reference)
    dist_info = get_video_info(distorted)
    
    if not ref_info or not dist_info:
        if verbose:
            print_key_value("ERROR", "Could not retrieve video information")
        return False
    
    if verbose:
        print()
        print(f"{'ATTRIBUTE':<13} {'REFERENCE':<14} {'DISTORTED':<14}")
        print(f"{'Resolution':<13} {ref_info['resolution']:<14} {dist_info['resolution']:<14}")
        print(f"{'Framerate':<13} {ref_info['fps']:.3f} fps{'':<4} {dist_info['fps']:.3f} fps{'':<4}")
        print(f"{'Frame count':<13} {ref_info['frame_count']:<14} {dist_info['frame_count']:<14}")
        print(f"{'Duration':<13} {format_duration(ref_info['duration']):<14} {format_duration(dist_info['duration']):<14}")
        print(f"{'Pixel format':<13} {ref_info['pix_fmt']:<14} {dist_info['pix_fmt']:<14}")
        print(f"{'Color range':<13} {ref_info['color_range']:<14} {dist_info['color_range']:<14}")
        print(f"{'File size':<13} {format_file_size(ref_info['file_size']):<14} {format_file_size(dist_info['file_size']):<14}")
        
    messages = [] 
    if ref_info['width'] != dist_info['width'] or ref_info['height'] != dist_info['height']:
        messages.append(("ERROR", "Resolution mismatch"))
    
    if ref_info['frame_count'] != dist_info['frame_count']:
        messages.append(("ERROR", "Frame count mismatch"))

    fps_tolerance = 0.001
    if abs(ref_info['fps'] - dist_info['fps']) > fps_tolerance:
        messages.append(("WARNING", "Framerate mismatch"))

    if ref_info['color_range'] != dist_info['color_range'] and ref_info['color_range'] != 'unknown' and dist_info['color_range'] != 'unknown':
        messages.append(("WARNING", "Color range mismatch"))
    
    if ref_info['pix_fmt'] != dist_info['pix_fmt']:
        messages.append(("WARNING", "Pixel format mismatch"))

    
    has_errors = any(level == "ERROR" for level, _ in messages)
    has_warnings = any(level == "WARNING" for level, _ in messages)
    if verbose:
        if len(messages) > 0:
            print()

        for level, msg in messages:
            print_key_value(level, msg)
        
    if has_errors:
        return False
    
    if verbose and has_warnings:
        print("This may affect results but analysis will continue...")
        
    return True


def run_analysis(mode, distorted, reference=None, output_dir=None, verbose=True):
    properties_match = True
    if mode in FR_MODES and reference is not None:
        properties_match = compare_video_properties(reference, distorted, verbose=verbose)

        if mode == 'check':
            return properties_match, None
        
        if not properties_match:
            return properties_match, None

    if mode in MODES['ffmpeg']:
        return properties_match, run_ffmpeg(reference, distorted, mode, output_dir, verbose=verbose)
    elif mode in MODES['cvqa']:
        return properties_match, run_cvqa(reference, distorted, mode, output_dir, verbose=verbose)
    elif mode in MODES['lpips']:
        return properties_match, run_lpips(reference, distorted, mode, output_dir, verbose=verbose)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    
    parser = argparse.ArgumentParser(description='Run video quality analysis comparing a distorted video against a reference video')
    parser.add_argument("-d", '--distorted', required=True, help='Distorted (compressed) video file or folder')
    parser.add_argument("-r", '--reference', help='Reference (original) video file or folder (required for FR methods)')
    parser.add_argument("-m", '--mode', choices=['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr', 'check', 'cvqa-nr', 'cvqa-nr-ms', 'cvqa-fr', 'cvqa-fr-ms', 'lpips'], default='vmaf4k-full')
    parser.add_argument('-o', '--output', nargs='?', const='.', help='Save output files. Optional: specify directory (default: same as distorted file)')
    parser.add_argument('-q', '--quiet', default=False, action='store_true', help='Enable quiet output')
    args = parser.parse_args()

    print_separator("STARTING VQCHECK")

    if os.path.isfile(args.distorted):
        distorted_files = [args.distorted, ]
    else:
        distorted_files = get_video_files(args.distorted)
    
    if args.reference:
        if os.path.isfile(args.reference):
            reference_files = [args.reference,]
        else:
            reference_files = get_video_files(args.reference)

    print_key_value("Mode", f"{args.mode}")
    print_key_value("Distorted", f"{args.distorted} ({len(distorted_files)})")
    if args.reference:
        print_key_value("Reference", f"{args.reference} ({len(reference_files)})")

    if args.mode in FR_MODES and not args.reference:
        print("ERROR: Reference video is required for the selected mode")
        return
    
    if args.mode in NR_MODES and args.reference:
        args.reference = None # Ignore reference for NR modes

    total_files = len(distorted_files)
    matching_properties = 0
    perfect_match = 0
    
    for distorted in distorted_files:
        reference = None
        if len(reference_files) == 1:
            reference = reference_files[0]
        elif len(reference_files) > 1:
            reference = find_reference_file(distorted, reference_files)

        if not reference and args.mode in FR_MODES:
            print(f"ERROR: No reference file found for {distorted}")
            continue
        
        output_dir = None
        if args.output is not None:
            if args.output == '.':
                output_dir = os.path.dirname(distorted)
            else:
                output_dir = args.output

        print_separator(f"VQCheck ({args.mode})", newline=True)
        print_key_value("Distorted", distorted)
        if reference:
            print_key_value("Reference", reference)
        properties_match, results = run_analysis(args.mode, distorted, reference, output_dir, verbose=not args.quiet)
        print_separator("SKIPPED (property mismatch)" if not properties_match else "")

        if properties_match:
            matching_properties += 1
            
        if results:
            if 'psnr' in args.mode:
                if results.get('psnr_avg', 0) == float('inf'):
                    perfect_match += 1
            elif 'lpips' in args.mode:
                if results.get('metadata', {}).get('mean_distance', 1) == 0:
                    perfect_match += 1
            else:
                if results.get('psnr', 0) >= 60: # == inf for VMAF tool
                    perfect_match += 1
    
    if total_files > 1:
        print_separator("SUMMARY", newline=True)
        print_key_value("Files Processed", str(total_files))
        if args.mode in FR_MODES:
            print_key_value("Matching Properties", str(matching_properties))
            print_key_value("Perfect Matches", str(perfect_match))
        print_separator()
    

if __name__ == "__main__":
    main()
