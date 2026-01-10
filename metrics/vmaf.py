import os
import tempfile
import subprocess
from datetime import datetime
import json

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line, is_quiet, transcode_video

def check_vmaf(rebuild=False):
    """Check if VMAF command is available."""
    try:
        result = subprocess.run(['vmaf', '-v'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("VMAF binary test run failed")

    except Exception as e:
        print(e)
        print_line(f"ERROR: VMAF not found. Download or build it and add it to PATH", force=True)
        return False

    return True


def run_vmaf(mode, distorted, reference, scale=None, fps=None, output_dir=None, temp_dir=None):

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='vqcheck_vmaf_')

    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        
        if os.path.exists(output_file) and not __name__ == "__main__":
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    start_time = datetime.now()

    try:
        print_line("\nRESULTS")
        print_key_value("Start Time", ts(start_time))
        reference_temp_path = os.path.join(temp_dir, 'ref', f'{os.path.basename(reference)}.y4m')
        distorted_temp_path = os.path.join(temp_dir, 'dis', f'{os.path.basename(distorted)}.y4m')
        if not os.path.exists(os.path.dirname(reference_temp_path)):
            transcode_video(reference, reference_temp_path, format='rawvideo', scale=scale)
        transcode_video(distorted, distorted_temp_path, format='rawvideo', scale=scale)

        cmd = ['vmaf',
               '--reference', reference_temp_path,
               '--distorted', distorted_temp_path,
               '--output', output_file,
               '--json',
        ]
        cmd.extend(get_arguments(mode))

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")

        results = parse_vmaf_results(output_file, distorted, reference)
        if results:
            print_key_value("VMAF", f"{results['vmaf']:.2f}")
            if 'full' in mode:
                print_key_value("VMAF (neg)", f"{results['vmaf_neg']:.2f}")
            print_key_value("PSNR", f"{results['psnr']:.2f} dB")
            print_key_value("PSNR Y", f"{results['psnr_y']:.2f} dB")
            print_key_value("PSNR CB", f"{results['psnr_cb']:.2f} dB")
            print_key_value("PSNR CR", f"{results['psnr_cr']:.2f} dB")
            print_key_value("PSNR HVS", f"{results['psnr_hvs']:.2f} dB")
            print_key_value("SSIM", f"{results['ssim']:.4f}")
            print_key_value("MS-SSIM", f"{results['ms_ssim']:.4f}")

            if is_quiet():
                print_line(f"VMAF ({analysis_duration.total_seconds():.0f}s) | {results['vmaf']:.2f} | {os.path.basename(distorted)}", force=True)

        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)

        return results

    except Exception as e:
        print_line(" ".join(cmd), force=True)
        print_line(str(e), force=True)
        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        return None

def parse_vmaf_results(output_file, distorted=None, reference=None):
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        pooled_metrics = data.get('pooled_metrics', {})
        vmaf_score = pooled_metrics.get('vmaf', {}).get('mean', 0)
        vmaf_neg_score = pooled_metrics.get('vmaf_neg', {}).get('mean', 0)
        psnr_y_score = pooled_metrics.get('psnr_y', {}).get('mean', 0)
        psnr_cb_score = pooled_metrics.get('psnr_cr', {}).get('mean', 0)
        psnr_cr_score = pooled_metrics.get('psnr_cb', {}).get('mean', 0)
        psnr_hvs_score = pooled_metrics.get('psnr_hvs', {}).get('mean', 0)
        ssim_score = pooled_metrics.get('float_ssim', {}).get('mean', 0)
        ms_ssim_score = pooled_metrics.get('float_ms_ssim', {}).get('mean', 0)
        
        return {
            'timestamp': ts(),
            'distorted': os.path.basename(distorted),
            'reference': os.path.basename(reference),
            'vmaf': vmaf_score,
            'vmaf_neg': vmaf_neg_score,
            'psnr': (6*psnr_y_score+psnr_cb_score+psnr_cr_score)/8,
            'psnr_y': psnr_y_score,
            'psnr_cb': psnr_cb_score,
            'psnr_cr': psnr_cr_score,
            'psnr_hvs': psnr_hvs_score,
            'ssim': ssim_score,
            'ms_ssim': ms_ssim_score
        }
    except Exception as e:
        print_line(f"Error parsing VMAF results: {e}", force=True)
        return None


def get_arguments(mode):
    args = []
    if '4k' in mode:
        args.extend(['--model', 'version=vmaf_4k_v0.6.1'])
        if 'full' in mode:
            args.extend(['--model', 'version=vmaf_4k_v0.6.1neg:name=vmaf_neg'])
    else:
        args.extend(['--model', 'version=vmaf_v0.6.1'])
        if 'full' in mode:
            args.extend(['--model', 'version=vmaf_v0.6.1neg:name=vmaf_neg'])

    args.extend(['--feature', 'psnr', '--feature', 'psnr_hvs', '--feature', 'float_ssim', '--feature', 'float_ms_ssim'])

    threads = os.cpu_count()
    threads = max(1, threads - min(4, int(threads * 0.2)))
    args.extend(['--threads', str(threads)])
    return args

