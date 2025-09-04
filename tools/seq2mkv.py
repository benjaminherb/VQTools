import os
import sys
import subprocess
import argparse
from pathlib import Path


IMAGE_PATTERNS = ["*.png", "*.tiff", "*.tif", "*.jpg", "*.jpeg", "*.exr"]


def has_image_files(folder_path: Path):
    return any(folder_path.glob(p) for p in IMAGE_PATTERNS)


def get_image_pattern(folder_path: Path):
    for pat in IMAGE_PATTERNS:
        if any(folder_path.glob(pat)):
            return str(folder_path / pat)
    return str(folder_path / "*.*")


def get_image_files(folder_path: Path):
    pattern = get_image_pattern(folder_path)
    parent = Path(pattern).parent
    pat = Path(pattern).name
    return sorted(parent.glob(pat))


def encode_sequence(folder_path: Path, output_path: Path, fps=60, scale=None, ffvhuff=False, pix_fmt=None):
    input_pattern = get_image_pattern(folder_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        input_pattern,
    ]

    if ffvhuff:
        cmd.extend(["-c:v", "ffvhuff"])
    else:
        cmd.extend(["-c:v", "libx265", "-x265-params", "lossless=1"])

    if pix_fmt:
        cmd.extend(["-pix_fmt", pix_fmt])
    cmd.extend(["-r", str(fps)])

    if scale:
        cmd.extend(["-vf", f"scale={scale}:param0=5", "-sws_flags", "lanczos+accurate_rnd+bitexact"])

    cmd.append(str(output_path))

    try:
        print(f"Encoding {folder_path.name} -> {output_path.name}")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully encoded {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error encoding {folder_path.name}:")
        print(f"Return code: {e.returncode}")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Encode image sequences to MKV using FFmpeg")
    parser.add_argument("root_dir", help="Root directory containing folders with image sequences")
    parser.add_argument("--fps", type=int, default=60, help="Output framerate (default: 60)")
    parser.add_argument("--scale", nargs='?', const='3840:2160', default=None,
                        help="Scale the output to W:H using lanczos. If given without value defaults to 3840:2160.")
    parser.add_argument("--ffvhuff", default=False, action="store_true", help="Use ffvhuff instead of h265")
    parser.add_argument("--pix-fmt", default=None, help="Pixel format to set for ffmpeg")
    parser.add_argument("--output-dir", help="Output directory (default: <root>/mkv)")
    parser.add_argument("--dryrun", action="store_true", help="Do not run ffmpeg")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: Root directory '{root_path}' is not a directory or does not exist.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else root_path / "mkv"
    output_dir.mkdir(parents=True, exist_ok=True)

    folders_with_images = [p for p in root_path.iterdir() if p.is_dir() and has_image_files(p)]

    if not folders_with_images:
        print(f"No folders with image sequences found in '{root_path}'")
        sys.exit(0)

    print(f"Found {len(folders_with_images)} folders with image sequences:")
    for folder in folders_with_images:
        files = get_image_files(folder)
        ext = files[0].suffix.lower() if files else ""
        print(f"  - {folder.name} ({len(files)} files, ext: {ext})")

    scale_value = args.scale

    print("\nStarting encoding")
    success_count = 0

    for folder in folders_with_images:
        output_path = output_dir / f"{folder.name}.mkv"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path} (use --overwrite to replace)")
            continue

        # use explicit pixel format if provided, otherwise omit the -pix_fmt flag
        pix = args.pix_fmt

        if not args.dryrun:
            if encode_sequence(folder, output_path, args.fps, scale_value, args.ffvhuff, pix):
                success_count += 1
        print("-" * 50)

    print(f"\nEncoding complete: {success_count}/{len(folders_with_images)} successful")


if __name__ == "__main__":
    main()