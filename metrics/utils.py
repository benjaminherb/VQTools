import os
import json
from datetime import datetime


## ------ File ------ ##


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


def save_json(data, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {output_file}: {e}")


## ------ Format ------ ##


def get_output_filename(distorted, mode, output_dir=None):
    base_name = os.path.splitext(os.path.basename(distorted))[0]
    
    if output_dir is None:
        output_dir = os.path.dirname(distorted)
    
    if 'psnr' in mode:
        return os.path.join(output_dir, f"{base_name}.psnr.json")
    elif mode.startswith('cvqa'):
        return os.path.join(output_dir, f"{base_name}.{mode}.json")
    else:
        return os.path.join(output_dir, f"{base_name}.vmaf.json")


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


def print_separator(text=None, width=40, char='=', newline=False):
    """Print a separator line with optional centered text."""
    if text:
        text_len = len(text)
        if text_len >= width - 4:  # check if its too long
            print(text)
            return
        
        padding = (width - text_len - 2) // 2  # -2 for spaces around text
        left_pad = char * padding
        right_pad = char * (width - padding - text_len - 2)
        separator = f"{left_pad} {text} {right_pad}"
    else:
        separator = char * width
    
    if newline:
        print()
    print(separator)


def print_key_value(key, value, width=40):
    """Print a key-value pair with left-aligned key and right-aligned value"""
    key_str = f"{key}:"
    total_needed = len(key_str) + len(str(value)) + 1 
    
    if total_needed <= width: # right align
        spaces_needed = width - len(key_str) - len(str(value))
        formatted_line = f"{key_str}{' ' * spaces_needed}{value}"
    else:
        formatted_line = f"{key_str} {value}"
    
    print(formatted_line)


def ts(time=None):
    if time:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")