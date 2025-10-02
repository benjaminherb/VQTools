import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import pyiqa
import cv2
import torch
import numpy as np
from statistics import mean
import time
import gc

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line, get_video_info


def check_pyiqa(mode):
    try:
        metric = pyiqa.create_metric(mode, as_loss=False, device='cpu')  # download if needed
    except Exception as e:
        print_line(f"Error checking PyIQA for model {mode}: {e}", force=True)
        return False
    return True

def _preprocess_frame(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # Add batch dimension
    return frame_tensor

def _process_frames_streaming(video_path, reference_path, metric, device, stride=1):
    """Process frames from video one by one"""
    is_fr = reference_path is not None
    frame_scores = {}
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        cap_ref = None
        if is_fr:
            cap_ref = cv2.VideoCapture(reference_path)
            if not cap_ref.isOpened():
                raise Exception(f"Could not open reference video: {reference_path}")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ret, frame_ref = None, None
            if is_fr:
                ret, frame_ref = cap_ref.read()
                
            if frame_number % stride == stride // 2:
                frame_tensor = _preprocess_frame(frame, device)
                
                try:
                    if is_fr:
                        frame_tensor_ref = _preprocess_frame(frame_ref, device)
                        score = float(metric(frame_tensor, frame_tensor_ref).cpu().item())
                    else:   
                        score = float(metric(frame_tensor).cpu().item())
                    frame_scores[frame_number] = score
                except Exception as e:
                    print_line(f"Error on frame {frame_number} (SKIPPING): {e}", force=True)


            frame_number += 1

        cap.release()
        
        if not frame_scores:
            raise Exception(f"No frames processed from video: {video_path}")
            
        return frame_scores
        
    except Exception as e:
        raise Exception(f"Error processing frames: {e}")


def run_pyiqa(mode, distorted, reference, output_dir=None):
    """Run PyIQA video quality assessment."""

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
        if str(device) == 'mps' and mode.lower() == 'niqe':
            device = 'cpu' # NIQE has issues with MPS and float64

        fps = get_video_info(distorted).get('fps', 60)
        stride = max(1, int(fps/2))

        metric = pyiqa.create_metric(mode, as_loss=False, device=device)
        result = _process_frames_streaming(distorted, reference, metric, device, stride=stride)

        frame_scores = list(result.values())

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
            'frame_numbers': list(result.keys())
            }

        if reference:
            results['reference'] = os.path.basename(reference) if reference else None

        results.update({
            'metric': mode,
            'mean_score': mean_score,
            'min_score': min_score,
            'max_score': max_score,
            'frame_scores': result,
            'stride': stride,
        })
        
        if output_file:
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: PyIQA evaluation failed: {e}", force=True)
        return None

