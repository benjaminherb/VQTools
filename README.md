## VQTools

This repository provides a lightweight command-line toolkit for running video quality metrics and related utilities. `vqcheck` offers an easy way to build and run several recent full-reference and no-reference models while providing a unified output format (JSON).

## Installation
To install the included tools run the following on either Linux or macOS:
```bash
git clone https://github.com/benjaminherb/VQTools
cd VQTools
chmod +x ./install.sh
./install.sh
```
This installs the tools in `~/.local/share/vqtools`.
An ffmpeg/ffprobe installation with libvmaf support is required for VMAF modes.

## VQCheck

Run a no-reference model on a single video:

```bash
vqcheck -m dover -d distorted.mp4
```

Run a full-reference model on a set of videos and save the results:

```bash
vqcheck -m lpips -d /path/to/distorted -r /path/to/references/ -o ./results/
```

#### Options:
- `-d, --distorted` (required) — Distorted video file or folder
- `-r, --reference` — Reference video file or folder (required for FR modes)
- `-m, --mode` — Which metric/mode to run (see table below).
- `-o, --output` — Optional: save output files (one `*.method.json` per video). If `-o` is passed without an argument, the output is written next to the distorted files.
- `-q, --quiet` — Reduce console output

#### Notes:
- If `--reference` is a folder, the script uses the closest filename match.
- If `psnr` is calculated, the script also returns whether the file and reference are a perfect match (e.g., to check for lossless errors).
- When running a method for the first time, its code and required files are pulled and built.
- MaxVQA seems to have issues with ffmpeg 8 but works with ffmpeg 7.
- The script tries to use GPU backends where available (MPS/Vulkan/CUDA), but will fall back to CPU

### Models

| Model | Type | Methods | Repository | Notes |
|---|---|---|---|---|
| Check | FR | `check` | | Metadata check to ensure metadata for distorted and reference match (resolution, framerate, framecount etc)|
| PSNR | FR | `psnr` | | Faster way to just calculate PSNR|
| VMAF | FR | `vmaf` `vmaf4k` `vmaf-full` `vmaf4k-full` | [VMAF](https://github.com/Netflix/vmaf) | Always includes SSIM/MS-SSIM and PSNR; `full` variants additionally include the NEG (no-enhancement gain) variants (slower).|
| LPIPS |FR| `lpips` | [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | [Paper](https://arxiv.org/abs/1801.03924) - frame-wise metric, uses `alex` as network |
| PyIQA | FR/NR | `musiq` | [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch) | [Musiq](https://arxiv.org/abs/2108.05997) — image quality metrics provided as mean and per-frame scores (sampled at 2 per second). |
| CVQA | FR/NR | `cvqa-nr` `cvqa-nr-ms` `cvqa-fr` `cvqa-fr-ms` | [CompressedVQA](https://github.com/sunwei925/CompressedVQA) | [Paper](https://arxiv.org/abs/2106.01111) |
| FastVQA |NR| `fastvqa` `fastervqa`| [FAST-VQA/FasterVQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) | [Paper](https://arxiv.org/abs/2210.05357) |
| UVQ |NR| `uvq` | [UVQ](https://github.com/google/uvq) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Rich_Features_for_Perceptual_Quality_Assessment_of_UGC_Videos_CVPR_2021_paper.html) - separate and combined scores for compression, distortion and content|
| Dover |NR| `dover` | [DOVER](https://github.com/QualityAssessment/DOVER) | [Paper](https://arxiv.org/abs/2211.04894) -  includes technical and aesthetic score |
| Q-Align |NR| `qalign` | [Q-Align](https://github.com/Q-Future/Q-Align) | [Paper](https://arxiv.org/abs/2312.17090) |
| MaxVQA | NR | `maxvqa` | [ExplainableVQA](https://github.com/VQAssessment/ExplainableVQA) | [Paper](https://arxiv.org/abs/2305.12726) - several scores for different quality factors |
| Cover | NR | `cover` | [COVER](https://github.com/taco-group/COVER) | [Paper](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf) — semantic, technical and aesthetic scores. |

## Other Tools

### seq2mkv
Encode image sequences (`png`, `jpeg`, `tiff`) located in subfolders under a root directory into MKV files using ffmpeg.
Options:
- `root_dir` (positional) — root directory containing folders with image sequences
- `--fps` — output framerate (default: 60)
- `--scale` — optional W:H scale (defaults to `3840:2160` when provided without value)
- `--ffvhuff` — use `ffvhuff` codec instead of H.265 (lossless)
- `--pix-fmt` — explicit pixel format to pass to ffmpeg
- `--output-dir` — output directory (default: `<root_dir>/mkv`)
- `--dryrun` — do not run ffmpeg; just show commands
- `--overwrite` — overwrite existing output files

Example:

```bash
seq2mkv /path/to/sequences --scale 3840:2160 --ffvhuff --output-dir ./mkv
```

### transcode
Transcode all video files in a directory to a standardized MKV (lossless) using either `ffvhuff` (for playback) or `h265` (for storage).

Options:
- `-i/--input` — input directory containing video files (required)
- `-o/--output` — output directory for converted files (required)
- `--codec` — one of `ffvhuff` or `h265` (default: `h265`)
- `--overwrite` — overwrite existing files in the output directory
- `--dryrun` — show ffmpeg commands without executing them

Example:

```bash
transcode -i ./videos -o ./converted --codec ffvhuff
```

### aggmet
Consolidate multiple metric JSON files into a single JSON report. The script scans a directory tree for `*.method.json` files and groups them by base video name.

Options:
- `--metrics-dir, -m` — directory containing individual metric JSON files (required)
- `--output-file, -o` — path to write consolidated JSON (required)
- `--existing-json, -e` — optional existing combined JSON to update/merge

Example:

```bash
aggmet -m ./metrics -o consolidated.json
```



## License and attribution

See each linked repository for license and citation details — this project provides only integration code.
