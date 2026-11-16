import os
from datetime import datetime
from statistics import mean

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, get_video_info, \
    is_quiet
from metrics.pyiqa import _process_frames_streaming


def _get_metric(mode):
    import piq
    if mode == 'haarpsi':
        return lambda x, y: piq.haarpsi(x, y, data_range=1.0, reduction='mean')
    raise ValueError(f"Unknown PIQ mode: {mode}")


def check_piq(mode):
    try:
        _get_metric(mode)
    except Exception as e:
        print_line(f"Error checking PIQ for metric {mode}: {e}", force=True)
        return False
    return True


def run_piq(mode, distorted, reference, output_dir=None, scale=None):
    """Run a frame-wise PIQ full-reference metric (sampled at 2 frames per second)."""

    output_file = None
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime.now()
    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))

    try:
        device = get_device()
        fps = get_video_info(distorted).get('fps', 60)
        stride = max(1, int(fps / 2))

        metric = _get_metric(mode)
        result = _process_frames_streaming(distorted, reference, metric, device, stride=stride, scale=scale)

        frame_scores = list(result.values())
        mean_score = mean(frame_scores)
        min_score = min(frame_scores)
        max_score = max(frame_scores)

        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")
        print_key_value("Frames Processed", len(frame_scores))
        print_key_value("Mean Score", f"{mean_score:.4f}")
        print_key_value("Min Score", f"{min_score:.4f}")
        print_key_value("Max Score", f"{max_score:.4f}")

        if is_quiet():
            print_line(f"{mode.upper()} ({analysis_duration.total_seconds():.0f}s) | {mean_score:.4f} | {os.path.basename(reference)} | {os.path.basename(distorted)}", force=True)

        results = {
            'timestamp': ts(),
            'distorted': os.path.basename(distorted),
            'reference': os.path.basename(reference),
            'frame_numbers': list(result.keys()),
            'metric': mode,
            'mean_score': mean_score,
            'min_score': min_score,
            'max_score': max_score,
            'frame_scores': result,
            'stride': stride,
        }

        if output_file:
            save_json(results, output_file)

        return results

    except Exception as e:
        print_line(f"ERROR: PIQ evaluation failed: {e}", force=True)
        return None
