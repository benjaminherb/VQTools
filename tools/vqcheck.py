#!/usr/bin/env python3
import sys
import subprocess
import os
import argparse
import cv2
import json
from metrics import run_lpips, run_ffmpeg, run_cvqa
from metrics.utils import get_video_files, find_reference_file, format_duration, format_file_size

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
            print("ERROR: Could not retrieve video information")
        return False
    
    if verbose:
        print()
        print(f"{'ATTRIBUTE':<15} {'REFERENCE':<15} {'DISTORTED':<15}")
        print(f"{'Resolution':<15} {ref_info['resolution']:<15} {dist_info['resolution']:<15}")
        print(f"{'Framerate':<15} {ref_info['fps']:.3f} fps{'':<5} {dist_info['fps']:.3f} fps{'':<5}")
        print(f"{'Frame count':<15} {ref_info['frame_count']:<15} {dist_info['frame_count']:<15}")
        print(f"{'Duration':<15} {format_duration(ref_info['duration']):<15} {format_duration(dist_info['duration']):<15}")
        print(f"{'Pixel format':<15} {ref_info['pix_fmt']:<15} {dist_info['pix_fmt']:<15}")
        print(f"{'Color range':<15} {ref_info['color_range']:<15} {dist_info['color_range']:<15}")
        print(f"{'File size':<15} {format_file_size(ref_info['file_size']):<15} {format_file_size(dist_info['file_size']):<15}")
        
    has_errors = False
    
    if ref_info['width'] != dist_info['width'] or ref_info['height'] != dist_info['height']:
        if verbose:
            print(f"ERROR: Resolution mismatch - Reference: {ref_info['width']}x{ref_info['height']} vs Distorted: {dist_info['width']}x{dist_info['height']}")
        has_errors = True
    
    if ref_info['frame_count'] != dist_info['frame_count']:
        if verbose:
            print(f"ERROR: Frame count mismatch - Reference: {ref_info['frame_count']} vs Distorted: {dist_info['frame_count']}")
        has_errors = True

    fps_tolerance = 0.001
    if abs(ref_info['fps'] - dist_info['fps']) > fps_tolerance:
        if verbose:
            print(f"Warning: Framerate mismatch - Reference: {ref_info['fps']:.3f}fps vs Distorted: {dist_info['fps']:.3f}fps")
        has_errors = True
    
    if ref_info['color_range'] != dist_info['color_range'] and ref_info['color_range'] != 'unknown' and dist_info['color_range'] != 'unknown':
        if verbose:
            print(f"WARNING: Color range mismatch - Reference: {ref_info['color_range']} vs Distorted: {dist_info['color_range']}")
            print("This may affect results but analysis will continue...")
    
    if ref_info['pix_fmt'] != dist_info['pix_fmt']:
        if verbose:
            print(f"WARNING: Pixel format mismatch - Reference: {ref_info['pix_fmt']} vs Distorted: {dist_info['pix_fmt']}")
            print("This may affect results but analysis will continue...")
    
    if has_errors:
        return False
        
    return True


def run_analysis(mode, distorted, reference=None, output_dir=None):
    properties_match = True
    if mode in FR_MODES and reference is not None:
        properties_match = compare_video_properties(reference, distorted)

        if mode == 'check':
            return properties_match, None
        
        if not properties_match:
            print(f"SKIPPING due to property mismatch!")
            return properties_match, None

    if mode in MODES['ffmpeg']:
        return properties_match, run_ffmpeg(reference, distorted, mode, output_dir)
    elif mode in MODES['cvqa']:
        return properties_match, run_cvqa(reference, distorted, mode, output_dir)
    elif mode in MODES['lpips']:
        return properties_match, run_lpips(reference, distorted, output_dir=output_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    
    parser = argparse.ArgumentParser(description='Run video quality analysis comparing a distorted video against a reference video')
    parser.add_argument("-d", '--distorted', required=True, help='Distorted (compressed) video file or folder')
    parser.add_argument("-r", '--reference', help='Reference (original) video file or folder (required for FR methods)')
    parser.add_argument("-m", '--mode', choices=['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr', 'check', 'cvqa-nr', 'cvqa-nr-ms', 'cvqa-fr', 'cvqa-fr-ms', 'lpips'], default='vmaf4k-full')
    parser.add_argument('--output', nargs='?', const='.', help='Save output files. Optional: specify directory (default: same as distorted file)')
    args = parser.parse_args()

    print(f"==== STARTING VQCHECK ====") 
    print(f"Mode: {args.mode}")
    print(f"Distorted: {args.distorted}")

    if os.path.isfile(args.distorted):
        distorted_files = [args.distorted, ]
    else:
        distorted_files = get_video_files(args.distorted)
    
    if args.reference:
        print(f"Reference: {args.reference}")

        if os.path.isfile(args.reference):
            reference_files = [args.reference,]
        else:
            reference_files = get_video_files(args.reference)

    if args.mode in FR_MODES and not args.reference:
        print("ERROR: Reference video is required for the selected mode")
        return
    
    if args.mode in NR_MODES and args.reference:
        args.reference = None # Ignore reference for NR modes

    total_files = len(distorted_files)
    matching_properties = 0
    perfect_match = 0
    print(distorted_files)
    
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

        print("\n==== VQCheck ====")
        print(f"Distorted: {distorted}")
        if reference:
            print(f"Reference: {reference}")
        properties_match, results = run_analysis(args.mode, distorted, reference, output_dir)
        print("===================")
        
        if properties_match:
            matching_properties += 1
            
        if results:
            if 'psnr' in args.mode:
                if results.get('psnr_avg', 0) == float('inf'):
                    perfect_match += 1
            else:
                if results.get('psnr', 0) >= 60: # == inf for VMAF tool
                    perfect_match += 1
    
    if total_files > 1:
        print(f"\n==== SUMMARY ====")
        print(f"Total files processed: {total_files}")
        if args.mode in FR_MODES:
            print(f"Files with matching properties: {matching_properties}")
            print(f"Files with infinite PSNR (perfect match): {perfect_match}")
        print("=================")
    

if __name__ == "__main__":
    main()
