#!/usr/bin/env python3
import sys
import subprocess
import os
import argparse
import cv2
import json
from datetime import datetime

def get_video_files(dir):
    video_extensions = ('.mp4', '.mkv', '.mov')
    video_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not os.path.splitext(file)[1].lower() in video_extensions:
                continue
            if file.startswith('.'):
                continue

            video_files.append(os.path.join(root, file))
    return video_files

def find_reference_file(distorted_file, reference_files):
    target_name = os.path.splitext(os.path.basename(distorted_file))[0]
    best_match = None
    max_matching_chars = 0
    for reference in reference_files:
        reference_name = os.path.splitext(os.path.basename(reference))[0]
        matching_chars = 0
        min_length = min(len(target_name), len(reference_name))
        for i in range(min_length):
            if target_name[i].lower() == reference_name[i].lower():
                matching_chars += 1
            else:
                break
        
        if matching_chars > max_matching_chars:
            max_matching_chars = matching_chars
            best_match = reference
    
    return best_match

def get_video_info(video_path):
    """Get video information using ffprobe and OpenCV"""
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
        
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
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

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def format_file_size(size_bytes):
    if size_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def compare_video_properties(reference, distorted):
    print("Checking video properties...")
    
    ref_info = get_video_info(reference)
    dist_info = get_video_info(distorted)
    
    if not ref_info or not dist_info:
        print("ERROR: Could not retrieve video information")
        return False
    
    print("\n=== VIDEO INFORMATION ===")
    print(f"Reference: {os.path.basename(reference)}")
    print(f"  Resolution: {ref_info['width']}x{ref_info['height']}")
    print(f"  Framerate: {ref_info['fps']:.3f}fps")
    print(f"  Frame Count: {ref_info['frame_count']}")
    print(f"  Duration: {format_duration(ref_info['duration'])}")
    print(f"  Pixel Format: {ref_info['pix_fmt']}")
    print(f"  Color Range: {ref_info['color_range']}")
    print(f"  File Size: {format_file_size(ref_info['file_size'])}")
    
    print(f"\nDistorted: {os.path.basename(distorted)}")
    print(f"  Resolution: {dist_info['width']}x{dist_info['height']}")
    print(f"  Framerate: {dist_info['fps']:.3f}fps")
    print(f"  Frame Count: {dist_info['frame_count']}")
    print(f"  Duration: {format_duration(dist_info['duration'])}")
    print(f"  Pixel Format: {dist_info['pix_fmt']}")
    print(f"  Color Range: {dist_info['color_range']}")
    print(f"  File Size: {format_file_size(dist_info['file_size'])}")
    print("========================\n")
    
    has_errors = False
    
    if ref_info['width'] != dist_info['width'] or ref_info['height'] != dist_info['height']:
        print(f"ERROR: Resolution mismatch - Reference: {ref_info['width']}x{ref_info['height']} vs Distorted: {dist_info['width']}x{dist_info['height']}")
        has_errors = True
    
    if ref_info['frame_count'] != dist_info['frame_count']:
        print(f"ERROR: Frame count mismatch - Reference: {ref_info['frame_count']} vs Distorted: {dist_info['frame_count']}")
        has_errors = True

    fps_tolerance = 0.001
    if abs(ref_info['fps'] - dist_info['fps']) > fps_tolerance:
        print(f"Warning: Framerate mismatch - Reference: {ref_info['fps']:.3f}fps vs Distorted: {dist_info['fps']:.3f}fps")
        has_errors = True
    
    if ref_info['color_range'] != dist_info['color_range'] and ref_info['color_range'] != 'unknown' and dist_info['color_range'] != 'unknown':
        print(f"WARNING: Color range mismatch - Reference: {ref_info['color_range']} vs Distorted: {dist_info['color_range']}")
        print("This may affect VMAF results but analysis will continue...")
    
    if ref_info['pix_fmt'] != dist_info['pix_fmt']:
        print(f"WARNING: Pixel format mismatch - Reference: {ref_info['pix_fmt']} vs Distorted: {dist_info['pix_fmt']}")
        print("This may affect VMAF results but analysis will continue...")
    
    if has_errors:
        return False
        
    return True

def get_lavfi(mode, output_file='-'):
    mode = mode.lower()
    if 'vmaf' in mode:
        if '4k' in mode :
            model_name, model_neg_name = "vmaf_4k_v0.6.1", "vmaf_4k_v0.6.1neg"
        else:
            model_name_model_neg_mae =  "vmaf_v0.6.1", "vmaf_v0.6.1neg"

        if 'full' in mode:
            lavfi = f"libvmaf='model=version={model_name}\\:name=vmaf|version={model_neg_name}\\:name=vmaf_neg:feature=name=psnr|name=float_ssim|name=float_ms_ssim:log_fmt=json:n_threads=16:log_path={output_file}'"
        else:
            lavfi = f"libvmaf='model=version={model_name}\\:name=vmaf:feature=name=psnr|name=float_ssim|name=float_ms_ssim:log_fmt=json:n_threads=16:log_path={output_file}'"

    elif 'psnr' in mode:
        lavfi = f"psnr=stats_file{output_file}"

    return lavfi


def parse_vmaf_results(output_file):
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        pooled_metrics = data.get('pooled_metrics', {})
        vmaf_score = pooled_metrics.get('vmaf', {}).get('mean', 0)
        vmaf_neg_score = pooled_metrics.get('vmaf_neg', {}).get('mean', 0)
        psnr_score = pooled_metrics.get('psnr', {}).get('mean', 0)
        ssim_score = pooled_metrics.get('float_ssim', {}).get('mean', 0)
        ms_ssim_score = pooled_metrics.get('float_ms_ssim', {}).get('mean', 0)
        
        return {
            'vmaf': vmaf_score,
            'vmaf_neg': vmaf_neg_score,
            'psnr': psnr_score,
            'ssim': ssim_score,
            'ms_ssim': ms_ssim_score
        }
    except Exception as e:
        print(f"Error parsing VMAF results: {e}")
        return None

def run_vmaf_analysis(reference, distorted, mode, check=False, output=False):

    output_file = f"{distorted[:-4]}.vmaf.json"
    if os.path.exists(output_file) and not __name__ == "__main__":
        print(f"{output_file} exists already - SKIPPING!")
        return
    
    if not compare_video_properties(reference, distorted):
        print(f"{output_file} SKIPPING due to property mismatch!")
        return
    
    if output:
        lavfi = get_lavfi(mode, output_file)
    else:
        lavfi = get_lavfi(mode)
    print(f"Using mode: {mode}")

    cmd = [
        'ffmpeg',
        '-i', distorted,
        '-i', reference,  
        '-lavfi', lavfi,
        '-f', 'null',
        '-'
    ]
    
    start_time = datetime.now()
    print(f"Starting Analysis {start_time}")
    print(f"Reference: {os.path.basename(reference)} / Distorted: {os.path.basename(distorted)}")
    
    if check:
        return

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print(f"Analysis completed in {analysis_duration}")
        print(f"Results saved to: {output_file}")
        
        results = parse_vmaf_results(output_file)
        if results:
            print("\n=== ANALYSIS RESULTS ===")
            print(f"VMAF:       {results['vmaf']:.2f}")
            print(f"VMAF (neg): {results['vmaf']:.2f}")
            print(f"PSNR:       {results['psnr']:.2f} dB")
            print(f"SSIM:       {results['ssim']:.4f}")
            print(f"MS-SSIM:    {results['ms_ssim']:.4f}")
            print("========================")

    except Exception as e:
        print(" ".join(cmd))
        print(e)


def main():
    
    parser = argparse.ArgumentParser(description='Run VMAF analysis comparing a distorted video against a reference video')
    parser.add_argument("-r", '--reference', required=True, help='Reference (original) video file or folder')
    parser.add_argument("-d", '--distorted', required=True, help='Distorted (compressed) video file or folder')
    parser.add_argument("-m", '--mode', choices=['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr'], default='vmaf4k-full')
    parser.add_argument('--check', action="store_true", help='Dont run VMAF, just do the precheck')
    args = parser.parse_args()

    if os.path.isfile(args.reference):
        reference_files = [args.reference,]
    else:
        reference_files = get_video_files(args.reference)

    if os.path.isfile(args.distorted):
        distorted_files = [args.distorted, ]
    else:
        distorted_files = get_video_files(args.distorted)

    for distorted in distorted_files:
        reference = None
        if len(reference_files) == 1:
            reference = reference_files[0]
        else:
            reference = find_reference_file(distorted, reference_files)

        if not reference:
            print(f"ERROR: No reference file found for {distorted}")
            continue
        
        run_vmaf_analysis(reference, distorted, args.mode, args.check)
    
if __name__ == "__main__":
    main()
