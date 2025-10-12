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


def run_ffprobe(path: Path) -> dict:
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
        key_map = {"width": "width", "height": "height", "r_frame_rate": "frame_rate", 
                   "pix_fmt": "pixel_format", "color_range": "color_range", "codec_name:": "codec"}
        for k, v in key_map.items():
            if k in raw_data["streams"][0]:
                data[v] = raw_data["streams"][0][k] 
        
        if "tags" in raw_data["streams"][0]:
            tags = raw_data["streams"][0]["tags"]
            if "ENCODER" in tags:
                data["encoder"] = tags["ENCODER"]
    
    if "format" in raw_data:
        key_map = {"format_name": "format", "size": "file_size", "bit_rate": "bit_rate", "duration": "duration"}
        for k, v in key_map.items():
            if k in raw_data["format"]:
                data[v] = raw_data["format"][k]

    # convert to numeric types where possible 
    for k in ["width", "height", "duration", "file_size", "bit_rate", "frame_rate"]:
        if k not in data:
            continue
        try:
            if k in ["width", "height", "file_size", "bit_rate"]:
                data[k] = int(data[k])
            elif k == "frame_rate":
                data[k] = eval(data[k])
            else:
                data[k] = float(data[k])
        except ValueError:
            pass
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from media files using ffprobe")
    parser.add_argument("--input_dir", "-i", required=True, type=Path, help="Input folder with media files")
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Output folder to write per-file metadata JSON")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--ext", "-e", action="append", help="Additional extensions to include (e.g. .avi). Can be used multiple times")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder does not exist or is not a directory: {input_dir}")
        return

    exts = set(VIDEO_EXTS)
    if args.ext:
        for e in args.ext:
            exts.add(e if e.startswith(".") else f".{e}")

    files = list(collect_files(input_dir, args.recursive, exts))
    if not files:
        print("No media files found.")
        return

    failed = []
    success = []
    for p in tqdm(files, desc="Extracting metadata"):
        try:
            data = run_ffprobe(p)
            rel = p.relative_to(input_dir)
            out_path = output_dir.joinpath(rel).with_suffix(".meta.json")
            write_json(out_path, data)
            success.append(p)
        except Exception as e:
            print(f"Error processing {p} (SKIPPING): {e}")
            failed.append({"file": str(p), "error": str(e)})

    print()
    print(f"Processed: {len(success)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f" - {f['file']}: {f['error']}")


if __name__ == "__main__":
    main()
# ...existing code...