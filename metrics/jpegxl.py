import os
import subprocess
import tempfile
from datetime import datetime
from statistics import mean

from metrics.utils import get_output_filename, save_json, print_key_value, ts, print_line

def get_command(metric):
    if metric == 'ssimulacra2':
        return 'ssimulacra2'
    if metric == 'butteraugli':
        return 'butteraugli_main'
    raise ValueError(f"Unsupported JPEG XL metric: {metric}")

def check_jpegxl(metric):
    metric_cmd = get_command(metric)
    try:
        pass
        result = subprocess.run([metric_cmd], capture_output=True, text=True)
        if 'Usage' not in result.stdout and 'Usage' not in result.stderr: # some tools print to stderr
            print_line(f"{metric_cmd} did not work, please install JPEG XL tools (eg. brew install jpeg-xl)", force=True)
            return False

    except Exception as e:
        print_line(f"Error checking {metric}: {e}", force=True)
        return False

    return True


def _extract_frames(video_path, temp_dir, fps=2):
    cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", os.path.join(temp_dir.name, "frame_%06d.png")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_files = sorted([os.path.join(temp_dir.name, f) for f in os.listdir(temp_dir.name) if f.endswith('.png')])
    return frame_files 


def _process_frames(distorted_path, reference_path, metric):
    try:
        dis_temp_dir = tempfile.TemporaryDirectory()
        ref_temp_dir = tempfile.TemporaryDirectory()
        distorted_frames = _extract_frames(distorted_path, dis_temp_dir, fps=2)
        reference_frames = _extract_frames(reference_path, ref_temp_dir, fps=2)
    except Exception as e:
        print_line(f"Error extracting frames: {e}", force=True)
        return []

    if len(distorted_frames) != len(reference_frames):
        print_line(f"Frame count mismatch between distorted and reference videos.", force=True)
        return []

    frame_scores = []
    metric_cmd = get_command(metric)
    try:
        for dist_frame, ref_frame in zip(distorted_frames, reference_frames):

            cmd = [metric_cmd,  ref_frame, dist_frame]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            score = float(output.split()[0])
            frame_scores.append(score)
    except Exception as e:
        print_line(f"Error processing frames with {metric}: {e}", force=True)
        return []

    return frame_scores


def run_jpegxl_metric(metric, distorted, reference, output_dir=None):
    """Run JPEG XL metric."""

    output_file = None
    if output_dir is not None:
        output_file = get_output_filename(distorted, metric, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime.now()
    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))
    
    try:
        frame_scores = _process_frames(distorted, reference, metric)

        if not frame_scores:
            raise Exception("No frame scores computed.")

        mean_score = mean(frame_scores)
        min_score = min(frame_scores)
        max_score = max(frame_scores)
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time
        
        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        print_key_value("Frames Processed", len(frame_scores))
        print_key_value("Mean Score", f"{mean_score:.4f}", force=True)
        print_key_value("Min Score", f"{min_score:.4f}")
        print_key_value("Max Score", f"{max_score:.4f}")
        
        results = {
            'timestamp': ts(),
            'distorted': os.path.basename(distorted),
            }

        if reference:
            results['reference'] = os.path.basename(reference) if reference else None

        results.update({
            'metric': metric,
            'mean_score': mean_score,
            'min_score': min_score,
            'max_score': max_score,
            'frame_scores': frame_scores,
            'fps': 2,
        })
        
        if output_file:
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: {metric} evaluation failed: {e}", force=True)
        return None

