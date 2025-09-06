import os
import json
import subprocess
from datetime import datetime
import cv2


## ------ Video ------ ##


def get_frame_count_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count



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

## ------ Docker ------ ##


def check_docker():
    """Check if Docker is installed and the daemon is running."""
    try:
        # Check if docker command exists
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False
            
        # Check if daemon is running
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
            
        print("ERROR: Docker daemon is not running")
        return False
        
    except FileNotFoundError:
        print("ERROR: Docker is not installed")
        return False


def build_docker_image(image_name, source_path):
    """Build the Docker image if it doesn't exist."""
    
    # Check if image already exists
    result = subprocess.run(['docker', 'images', '-q', image_name], capture_output=True, text=True)
    if result.stdout.strip():
        return True

    print("Building Docker image...")
    result = subprocess.run(['docker', 'build',  '--rm', '-t', image_name, source_path], 
                            capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Docker image built successfully")
        return True
    else:
        print(f"ERROR: Failed to build Docker image: {result.stderr}")
        return False


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
    elif 'vmaf' in mode:
        return os.path.join(output_dir, f"{base_name}.vmaf.json")
    else:
        return os.path.join(output_dir, f"{base_name}.{mode}.json")


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