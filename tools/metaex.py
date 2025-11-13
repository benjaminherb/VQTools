import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm


VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts", ".m2ts", ".mpeg", ".mpg"
}


def is_media_file(p: Path, exts: List[str]) -> bool:
    return p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")


import subprocess
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_video_frames(video_path: Path) -> Tuple[List[Dict], Dict]:
    """ Extract frame information and aggregate statistics from a video file."""

    cmd = [
        'ffprobe',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries', 'frame=pts_time,pict_type,pkt_size',
        '-of', 'json',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    frames = []
    type_counts = defaultdict(int)
    type_sizes = defaultdict(list)
    for t in ['I', 'P', 'B']:
        type_sizes[t] = []
        type_counts[t] = 0
    
    for frame_num, frame in enumerate(data['frames']):
        frame_info = {
            'frame': frame_num,
            'pts': float(frame['pts_time']) if frame['pts_time'] != 'N/A' else None,
            'type': frame['pict_type'],
            'size': int(frame['pkt_size'])
        }
        frames.append(frame_info)
        
        pict_type = frame['pict_type']
        type_counts[pict_type] += 1
        type_sizes[pict_type].append(int(frame['pkt_size']))
    
    has_i = type_counts.get('I', 0) > 0
    has_p = type_counts.get('P', 0) > 0
    has_b = type_counts.get('B', 0) > 0

    if has_i and not has_p and not has_b:
        stream_type = 'I'
    elif has_i and has_p and not has_b:
        stream_type = 'IP'
    elif has_i and has_p and has_b:
        stream_type = 'IPB'
    elif has_i and has_b and not has_p:
        stream_type = 'IB'
    else:
        stream_type = ''
    
    stats = {
        'stream_type': stream_type,
        'frame_count': len(frames),
    }

    for t in ['I', 'P', 'B']:
        stats.update({
            f'{t.lower()}_frame_count': type_counts[t],
            f'{t.lower()}_frame_mean_size': sum(type_sizes[t]) / max(len(type_sizes[t]), 1),
            f'{t.lower()}_frame_total_size': sum(type_sizes[t])
        })
    
    return frames, stats


def run_ffprobe(path: Path, extrac_frame_data: bool=False) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v:0",
        str(path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed with code {proc.returncode}")

    data = {"file": path.name, "name": path.stem}
    raw_data = json.loads(proc.stdout)
    if "streams" in raw_data:
        key_map = {"width": "width", "height": "height", "r_frame_rate": "framerate", 
                   "pix_fmt": "pixel_format", "bits_per_raw_sample": "bit_depth", 
                   "color_range": "color_range", "codec_name": "codec"}
        for k, v in key_map.items():
            if k in raw_data["streams"][0]:
                data[v] = raw_data["streams"][0][k] 
        
        if "tags" in raw_data["streams"][0]:
            tags = raw_data["streams"][0]["tags"]
            if "ENCODER" in tags:
                data["encoder"] = tags["ENCODER"]
    
    if "format" in raw_data:
        key_map = {"format_name": "format", "size": "filesize", "bit_rate": "bitrate", "duration": "duration"}
        for k, v in key_map.items():
            if k in raw_data["format"]:
                data[v] = raw_data["format"][k]

    # convert to numeric types where possible 
    for k in ["width", "height", "duration", "filesize", "bitrate", "framerate"]:
        if k not in data:
            continue
        try:
            if k in ["width", "height", "filesize", "bitrate"]:
                data[k] = int(data[k])
            elif k == "framerate":
                data[k] = eval(data[k])
            else:
                data[k] = float(data[k])
        except ValueError:
            pass
    
    if extrac_frame_data:
        frames, frame_stats = analyze_video_frames(str(path))
        data.update(frame_stats)
        data['frames'] = frames

    return data


def collect_files(root: Path, recursive: bool, exts: List[str]):
    if recursive:
        for p in root.rglob("*"):
            if is_media_file(p, exts):
                yield p
    else:
        for p in root.iterdir():
            if is_media_file(p, exts):
                yield p


def write_json(output_path: Path, data: dict):
    """ Write JSON data to file with pretty formatting (compact for frames)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        keys = list(data.keys())
        
        for i, key in enumerate(keys):
            value = data[key]
            is_last = i == len(keys) - 1
            
            if key == "frames":
                f.write(f'  "{key}": [\n')
                for j, frame in enumerate(value):
                    frame_str = json.dumps(frame, ensure_ascii=False)
                    comma = "" if j == len(value) - 1 else ","
                    f.write(f"    {frame_str}{comma}\n")
                f.write("  ]")
            else:
                value_str = json.dumps(value, ensure_ascii=False)
                f.write(f'  "{key}": {value_str}')
            
            if not is_last:
                f.write(",")
            f.write("\n")
        
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from media files using ffprobe")
    parser.add_argument("--input_path", "-i", required=True, type=Path, help="Input folder/file")
    parser.add_argument("--output_path", "-o", type=Path, help="Output folder to write per-file metadata JSON")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--frame_data", "-f", action="store_true", help="Include per-frame data in output JSON")
    parser.add_argument("--ext", "-e", action="append", help="Additional extensions to include (e.g. .avi). Can be used multiple times")
    args = parser.parse_args()

    save_to_file = args.output_path is not None
    input_path: Path = args.input_path

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return

    exts = set(VIDEO_EXTS)
    if args.ext:
        for e in args.ext:
            exts.add(e if e.startswith(".") else f".{e}")

    files = []
    input_dir = input_path.parent if input_path.is_file() else input_path
    if input_path.is_file():
        files = [input_path] if is_media_file(input_path, list(exts)) else []
    elif input_path.is_dir():
        files = list(collect_files(input_path, args.recursive, exts))

    if not files:
        print("No media files found.")
        return

    failed = []
    success = []
    for p in tqdm(files, desc="Extracting metadata", disable=not save_to_file):
        try:
            data = run_ffprobe(p, args.frame_data)
            rel = p.relative_to(input_dir)
            if save_to_file:
                out_path = args.output_path.joinpath(rel).with_suffix(".meta.json")
                write_json(out_path, data)
            else:
                if "frames" in data:
                    data.pop("frames")

                print(json.dumps(data, ensure_ascii=False, indent=2))

            success.append(p)
        except Exception as e:
            print(f"Error processing {p} (SKIPPING): {e}")
            failed.append({"file": str(p), "error": str(e)})

    if len(success) > 1 or failed: # dont print summary for single file
        print()
        print(f"Processed: {len(success)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for f in failed:
                print(f" - {f['file']}: {f['error']}")


if __name__ == "__main__":
    main()