#!/usr/bin/env python3
import sys
import subprocess
import os
import argparse
import cv2
import json
import tempfile
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
            print("This may affect VMAF results but analysis will continue...")
    
    if ref_info['pix_fmt'] != dist_info['pix_fmt']:
        if verbose:
            print(f"WARNING: Pixel format mismatch - Reference: {ref_info['pix_fmt']} vs Distorted: {dist_info['pix_fmt']}")
            print("This may affect VMAF results but analysis will continue...")
    
    if has_errors:
        return False
        
    return True


def get_output_filename(distorted, mode, output_dir=None):
    base_name = os.path.splitext(os.path.basename(distorted))[0]
    
    if output_dir is None:
        output_dir = os.path.dirname(distorted)
    
    if 'psnr' in mode:
        return os.path.join(output_dir, f"{base_name}.psnr.json")
    else:
        return os.path.join(output_dir, f"{base_name}.vmaf.json")


def get_lavfi(mode, output_file):
    mode = mode.lower()
    lavfi = ''
    if 'vmaf' in mode:
        if '4k' in mode :
            model_name, model_neg_name = "vmaf_4k_v0.6.1", "vmaf_4k_v0.6.1neg"
        else:
            model_name, model_neg_name =  "vmaf_v0.6.1", "vmaf_v0.6.1neg"

        if 'full' in mode:
            lavfi = f"libvmaf='model=version={model_name}\\:name=vmaf|version={model_neg_name}\\:name=vmaf_neg:feature=name=psnr|name=float_ssim|name=float_ms_ssim:log_fmt=json:n_threads=16:log_path={output_file}'"
        else:
            lavfi = f"libvmaf='model=version={model_name}\\:name=vmaf:feature=name=psnr|name=float_ssim|name=float_ms_ssim:log_fmt=json:n_threads=16:log_path={output_file}'"

    elif 'psnr' in mode:
        lavfi = f"psnr='stats_file={output_file}'"

    return lavfi


def parse_vmaf_results(output_file):
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        pooled_metrics = data.get('pooled_metrics', {})
        vmaf_score = pooled_metrics.get('vmaf', {}).get('mean', 0)
        vmaf_neg_score = pooled_metrics.get('vmaf_neg', {}).get('mean', 0)
        psnr_y_score = pooled_metrics.get('psnr_y', {}).get('mean', 0)
        psnr_cb_score = pooled_metrics.get('psnr_cr', {}).get('mean', 0)
        psnr_cr_score = pooled_metrics.get('psnr_cb', {}).get('mean', 0)
        ssim_score = pooled_metrics.get('float_ssim', {}).get('mean', 0)
        ms_ssim_score = pooled_metrics.get('float_ms_ssim', {}).get('mean', 0)
        
        return {
            'vmaf': vmaf_score,
            'vmaf_neg': vmaf_neg_score,
            'psnr': (6*psnr_y_score+psnr_cb_score+psnr_cr_score)/8,
            'psnr_y': psnr_y_score,
            'psnr_cb': psnr_cb_score,
            'psnr_cr': psnr_cr_score,
            'ssim': ssim_score,
            'ms_ssim': ms_ssim_score
        }
    except Exception as e:
        print(f"Error parsing VMAF results: {e}")
        return None


def parse_psnr_results(temp_output_file):
    try:
        with open(temp_output_file, 'r') as f:
            lines = f.readlines()
        
        frame_data = []
        mse_y_values = []
        mse_u_values = []
        mse_v_values = []
        mse_avg_values = []
        psnr_y_values = []
        psnr_u_values = []
        psnr_v_values = []
        psnr_avg_values = []
        
        for line in lines:
            if line.startswith('n:'):
                parts = line.strip().split()
                frame_entry = {}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if key == 'n':
                            frame_entry[key] = int(value)
                        else:
                            frame_entry[key] = float(value)
                
                frame_data.append(frame_entry)
                mse_y_values.append(frame_entry['mse_y'])
                mse_u_values.append(frame_entry['mse_u'])
                mse_v_values.append(frame_entry['mse_v'])
                mse_avg_values.append(frame_entry['mse_avg'])
                psnr_y_values.append(frame_entry['psnr_y'])
                psnr_u_values.append(frame_entry['psnr_u'])
                psnr_v_values.append(frame_entry['psnr_v'])
                psnr_avg_values.append(frame_entry['psnr_avg'])
        
        pooled_metrics = {
            'psnr_y': {'mean': sum(psnr_y_values) / len(psnr_y_values)},
            'psnr_u': {'mean': sum(psnr_u_values) / len(psnr_u_values)},
            'psnr_v': {'mean': sum(psnr_v_values) / len(psnr_v_values)},
            'psnr_avg': {'mean': sum(psnr_avg_values) / len(psnr_avg_values)},
            'mse_y': {'mean': sum(mse_y_values) / len(mse_y_values)},
            'mse_u': {'mean': sum(mse_u_values) / len(mse_u_values)},
            'mse_v': {'mean': sum(mse_v_values) / len(mse_v_values)},
            'mse_avg': {'mean': sum(mse_avg_values) / len(mse_avg_values)}
        }
        
        return {
            'frames': frame_data,
            'pooled_metrics': pooled_metrics
        }
        
    except Exception as e:
        print(f"Error parsing PSNR results: {e}")
        return None


def save_psnr_json(psnr_data, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(psnr_data, f, indent=2)
    except Exception as e:
        print(f"Error saving PSNR JSON: {e}")


def run_vmaf_analysis(reference, distorted, mode, check=False, output_dir=None):
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        
        if os.path.exists(output_file) and not __name__ == "__main__":
            print(f"{output_file} exists already - SKIPPING!")
            return True, None
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        if 'psnr' in mode:
            temp_fd, output_file = tempfile.mkstemp(suffix='.txt', prefix='psnr_')
            os.close(temp_fd)
        else:
            temp_fd, output_file = tempfile.mkstemp(suffix='.json', prefix='vmaf_')
            os.close(temp_fd)
    
    properties_match = compare_video_properties(reference, distorted)

    if check:
        return properties_match, None

    if not properties_match:
        print(f"SKIPPING due to property mismatch!")
        if not output_dir:
            os.unlink(output_file)
        return
    
    lavfi = get_lavfi(mode, output_file)

    cmd = [
        'ffmpeg',
        '-i', distorted,
        '-i', reference,  
        '-lavfi', lavfi,
        '-f', 'null',
        '-'
    ]
    
    start_time = datetime.now()

    try:
        print("\nRESULTS")
        print(f"Start Time: {start_time}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print(f"End Time:   {end_time}")
        print(f"Duration:   {analysis_duration}")
        
        results = None
        if 'psnr' in mode:
            psnr_data = parse_psnr_results(output_file)
            if psnr_data:
                pooled = psnr_data['pooled_metrics']
                print(f"PSNR Y:     {pooled['psnr_y']['mean']:.2f} dB")
                print(f"PSNR U:     {pooled['psnr_u']['mean']:.2f} dB")
                print(f"PSNR V:     {pooled['psnr_v']['mean']:.2f} dB")
                print(f"PSNR Avg:   {pooled['psnr_avg']['mean']:.2f} dB")
                print(f"MSE Avg:    {pooled['mse_avg']['mean']:.2f}")
                
                if output_dir is not None:
                    final_output_file = get_output_filename(distorted, mode, output_dir)
                    save_psnr_json(psnr_data, final_output_file)
                    print(f"\nResults saved to: {final_output_file}")
                
                results = {'psnr_avg': pooled['psnr_avg']['mean']}
        else:
            results = parse_vmaf_results(output_file)
            if results:
                print(f"VMAF:       {results['vmaf']:.2f}")
                if 'neg' in mode:
                    print(f"VMAF (neg): {results['vmaf_neg']:.2f}")
                print(f"PSNR:       {results['psnr']:.2f} dB")
                print(f"PSNR Y:     {results['psnr_y']:.2f} dB")
                print(f"PSNR CB:    {results['psnr_cb']:.2f} dB")
                print(f"PSNR CR:    {results['psnr_cr']:.2f} dB")
                print(f"SSIM:       {results['ssim']:.4f}")
                print(f"MS-SSIM:    {results['ms_ssim']:.4f}")
                
                if output_dir is not None:
                    print(f"\nResults saved to: {output_file}")
        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        
        return properties_match, results

    except Exception as e:
        print(" ".join(cmd))
        print(e)
        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        return True, None


def main():
    
    parser = argparse.ArgumentParser(description='Run VMAF analysis comparing a distorted video against a reference video')
    parser.add_argument("-r", '--reference', required=True, help='Reference (original) video file or folder')
    parser.add_argument("-d", '--distorted', required=True, help='Distorted (compressed) video file or folder')
    parser.add_argument("-m", '--mode', choices=['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr'], default='vmaf4k-full')
    parser.add_argument('--check', action="store_true", help='Dont run VMAF, just do the precheck')
    parser.add_argument('--output', nargs='?', const='.', help='Save output files. Optional: specify directory (default: same as distorted file)')
    args = parser.parse_args()

    print(f"==== STARTING VQCHECK ====") 
    print(f"Reference: {args.reference}")
    print(f"Distorted: {args.distorted}")
    print(f"Mode: {args.mode}")

    if os.path.isfile(args.reference):
        reference_files = [args.reference,]
    else:
        reference_files = get_video_files(args.reference)

    if os.path.isfile(args.distorted):
        distorted_files = [args.distorted, ]
    else:
        distorted_files = get_video_files(args.distorted)

    total_files = len(distorted_files)
    matching_properties = 0
    perfect_match = 0
    
    for distorted in distorted_files:
        reference = None
        if len(reference_files) == 1:
            reference = reference_files[0]
        else:
            reference = find_reference_file(distorted, reference_files)

        if not reference:
            print(f"ERROR: No reference file found for {distorted}")
            continue
        
        output_dir = None
        if args.output is not None:
            if args.output == '.':
                output_dir = os.path.dirname(distorted)
            else:
                output_dir = args.output

        print("\n==== VQCheck ====")
        print(f"Reference: {reference}")
        print(f"Distorted: {distorted}")
        properties_match, results = run_vmaf_analysis(reference, distorted, args.mode, args.check, output_dir)
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
        print(f"Files with matching properties: {matching_properties}")
        print(f"Files with infinite PSNR (perfect match): {perfect_match}")
        print("=================")
    

if __name__ == "__main__":
    main()
