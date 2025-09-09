import os
import json
from pathlib import Path
import subprocess
from datetime import datetime
import cv2
import torch


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
        print_line(f"Error running ffprobe on {video_path}: {e}", force=True)
        return None
    except json.JSONDecodeError as e:
        print_line(f"Error parsing ffprobe output for {video_path}: {e}", force=True)
        return None


## ------ Virtual Environment ------ ##


def create_venv(venv_path, python='python3.12', requirements=None, compile_decord=False):
    """Create a virtual environment at the specified path."""
    try:
        result = subprocess.run([python, '-m', 'venv', venv_path], check=True)
        if result.returncode != 0:
            print_line(f"ERROR: Failed to create virtual environment at {venv_path}", force=True)
            return False

        if compile_decord:
            decord_dir =  Path(__file__).parent.parent / 'decord' / 'python'
            run_in_venv(venv_path, ['python', 'setup.py', 'install'], work_dir=str(decord_dir))
            # decord_lib = Path(venv_path) / 'decord' /  'libdecord.dylib'
            # target_dir = Path(venv_path) / 'lib' / python / 'site-packages' / 'decord'
            # subprocess.run(['cp', str(decord_lib), str(target_dir)], check=True)

        if requirements:
            pip_path = os.path.join(venv_path, 'bin', 'pip')
            result = run_in_venv(venv_path, [pip_path, 'install', '-r', requirements])
            if result.returncode != 0:
                print_line(f"ERROR: Failed to install requirements: {result.stderr}", force=True)
                return False
        return True
    except subprocess.CalledProcessError as e:
        print_line(f"ERROR: Failed to create virtual environment: {e}", force=True)
        return False


def run_in_venv(venv_path, command, work_dir=None):
    """Run a command inside the specified virtual environment."""
    if work_dir is None:
        work_dir = os.getcwd()
    
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = venv_path
    env['PATH'] = f"{os.path.join(venv_path, 'bin')}:{env.get('PATH', '')}"
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(work_dir), env=env)
        return result
    except Exception as e:
        print_line(f"ERROR: Failed to run command in venv: {e}", force=True)
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

        print_line("ERROR: Docker daemon is not running", force=True)
        return False
        
    except FileNotFoundError:
        print_line("ERROR: Docker is not installed", force=True)
        return False


def build_docker_image(image_name, source_path):
    """Build the Docker image if it doesn't exist."""
    
    # Check if image already exists
    result = subprocess.run(['docker', 'images', '-q', image_name], capture_output=True, text=True)
    if result.stdout.strip():
        return True

    print_line("Building Docker image...", force=True)
    result = subprocess.run(['docker', 'build',  '--rm', '-t', image_name, source_path], 
                            capture_output=True, text=True)
    
    if result.returncode == 0:
        print_line("Docker image built successfully", force=True)
        return True
    else:
        print_line(f"ERROR: Failed to build Docker image: {result.stderr}", force=True)
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
        print_line(f"Error saving JSON to {output_file}: {e}", force=True)


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


## ------ Print ------ ##


_quiet_mode = False


def set_quiet_mode(quiet=True):
    """Set global quiet mode for printing."""
    global _quiet_mode
    _quiet_mode = quiet


def print_line(text=None, force=False):
    """Print a line of text unless in quiet mode, unless forced."""
    if _quiet_mode and not force:
        return

    if text is None:
        print()
    else:
        print(text)

def print_separator(text=None, width=40, char='=', newline=False, force=False):
    """Print a separator line with optional centered text."""
    if _quiet_mode and not force:
        return

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


def print_key_value(key, value, width=40, force=False):
    """Print a key-value pair with left-aligned key and right-aligned value"""
    if _quiet_mode and not force:
        return

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


## ------ PyTorch ------ ##


def get_device():
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.backends.metal.is_available():
        return torch.device('metal')
    else:
        return torch.device('cpu')


## ------ File Modification ------ ##


def _read_file_lines(filepath):
    """Read file and return list of lines."""
    try:
        with open(filepath, 'r') as f:
            return f.readlines()
    except Exception as e:
        print_line(f"Error reading {filepath}: {e}", force=True)
        return None

def _write_file_lines(filepath, lines):
    """Write list of lines to file."""
    try:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print_line(f"Error writing {filepath}: {e}", force=True)
        return False

def _normalize_content(content):
    """Normalize content to list of lines with proper newlines."""
    if isinstance(content, str):
        content = [content]
    
    normalized = []
    for line in content:
        if not line.endswith('\n'):
            line += '\n'
        normalized.append(line)
    return normalized

def _find_pattern_line(lines, pattern):
    """Find first line index containing pattern."""
    for i, line in enumerate(lines):
        if pattern in line:
            return i
    return None

def modify_file(filepath, modifications):
    """
    Universal file modification function.
    
    Supported operations:
        {'action': 'replace', 'line': 10, 'content': 'new line'}  # 0-indexed
        {'action': 'replace', 'pattern': 'text', 'content': 'new line'}  
        {'action': 'insert', 'line': 12, 'content': ['line1', 'line2']}  # 0-indexed
        {'action': 'delete', 'from': 5, 'to': 8}  # from/to can be line numbers or patterns, use -1 for end
    """
    lines = _read_file_lines(filepath)
    if lines is None:
        return False
    
    def resolve_line_number(value, lines):
        if isinstance(value, str):
            return _find_pattern_line(lines, value)
        elif value == -1:
            return len(lines) - 1  
        else:
            return value
    
    sorted_mods = []
    for mod in modifications:
        if mod['action'] == 'replace' and 'pattern' in mod:
            line_idx = _find_pattern_line(lines, mod['pattern'])
            if line_idx is not None:
                sorted_mods.append((line_idx, mod))
        elif mod['action'] == 'insert' and 'pattern' in mod:
            line_idx = _find_pattern_line(lines, mod['pattern'])
            if line_idx is not None:
                sorted_mods.append((line_idx, mod))
        elif mod['action'] == 'delete':
            from_line = resolve_line_number(mod['from'], lines)
            if from_line is not None:
                sorted_mods.append((from_line, mod))
        elif 'line' in mod:
            sorted_mods.append((mod['line'], mod))
    
    sorted_mods.sort(key=lambda x: x[0], reverse=True)
    
    try:
        for line_idx, mod in sorted_mods:
            action = mod['action']
            
            if action == 'replace':
                if 'content' in mod:
                    if 'pattern' in mod:
                        lines[line_idx] = lines[line_idx].replace(mod['pattern'], mod['content'])
                    else:
                        content = _normalize_content(mod['content'])
                        lines[line_idx:line_idx+1] = content
                    
            elif action == 'insert':
                if 'content' in mod:
                    content = _normalize_content(mod['content'])
                    lines[line_idx+1:line_idx+1] = content
                    
            elif action == 'delete':
                from_line = resolve_line_number(mod['from'], lines)
                to_line = resolve_line_number(mod.get('to', mod['from']), lines)
                
                if from_line is not None and to_line is not None:
                    if mod.get('to') == -1:
                        del lines[from_line:]
                    else:
                        del lines[from_line:to_line+1]
        
        return _write_file_lines(filepath, lines)
        
    except Exception as e:
        print_line(f"Error applying modifications to {filepath}: {e}", force=True)
        return False