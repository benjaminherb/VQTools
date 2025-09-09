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

from metrics.utils import get_output_filename, save_json, print_key_value, ts, get_device, print_line


def check_pyiqa(mode):
    try:
        metric = pyiqa.create_metric(mode, as_loss=False, device='cpu')  # download if needed
    except Exception as e:
        print_line(f"Error checking PyIQA for model {mode}: {e}", force=True)
        return False
    return True


def _process_frames_streaming(video_path, metric, device, max_frames=None, stride=1):
    """Process frames from video one by one"""
    frame_scores = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % stride == 0:
                start_time = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(device)  # Add batch dimension
                conversion_time = time.time() - start_time
                print_line(f"Frame {processed_count + 1} conversion time: {conversion_time:.4f}s")
                
                # Calculate metric score
                start_inference_time = time.time()
                score = metric(frame_tensor).cpu().numpy()[0]
                inference_time = time.time() - start_inference_time
                print_line(f"Frame {processed_count + 1} inference time: {inference_time:.4f}s")
                
                frame_scores.append(score)
                processed_count += 1
                
                if max_frames and processed_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        if not frame_scores:
            raise Exception(f"No frames processed from video: {video_path}")
            
        return frame_scores
        
    except Exception as e:
        raise Exception(f"Error processing frames: {e}")


def run_pyiqa(mode, distorted, reference, output_dir=None):
    """Run PyIQA video quality assessment."""

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

        metric = pyiqa.create_metric(mode, as_loss=False, device=device)
        frame_scores = _process_frames_streaming(distorted, metric, device, stride=1)
        
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
            'reference': os.path.basename(reference) if reference else None,
            'metric': mode,
            'mean_score': mean_score,
            'min_score': min_score,
            'max_score': max_score,
            'frames_processed': len(frame_scores),
            'frame_scores': frame_scores
        }
        
        if output_file.endswith('.json'):
            save_json(results, output_file)
        
        return results
        
    except Exception as e:
        print_line(f"ERROR: PyIQA evaluation failed: {e}", force=True)
        return None

