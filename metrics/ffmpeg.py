import os
import tempfile
import subprocess
from datetime import datetime
import json

from metrics.utils import get_output_filename, save_json


def run_ffmpeg(reference, distorted, mode, output_dir=None):

    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        
        if os.path.exists(output_file) and not __name__ == "__main__":
            print(f"{output_file} exists already - SKIPPING!")
            return None
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        temp_fd, output_file = tempfile.mkstemp(prefix='vqcheck_')
        os.close(temp_fd)
    
    cmd = [
        'ffmpeg',
        '-i', distorted,
        '-i', reference,  
        '-lavfi', get_lavfi(mode, output_file),
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
                    save_json(psnr_data, final_output_file)
                    print(f"\nResults saved to: {final_output_file}")
                
                results = {'psnr_avg': pooled['psnr_avg']['mean']}
        else: # VMAF
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

        return results

    except Exception as e:
        print(" ".join(cmd))
        print(e)
        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        return None


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
