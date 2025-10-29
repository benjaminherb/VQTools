import os
import argparse
from metrics.utils import get_video_files, find_reference_file, format_duration, format_file_size, print_separator, print_key_value, get_video_info, set_quiet_mode, print_line

MODES = {
    'ffmpeg': ['vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr'],
    'cvqa': ['cvqa-nr', 'cvqa-nr-ms', 'cvqa-fr', 'cvqa-fr-ms'],
    'lpips': ['lpips'],
    'dover': ['dover'],
    'cover': ['cover'],
    'uvq': ['uvq'],
    'maxvqa': ['maxvqa'],
    'pyiqa': ['musiq', 'brisque', 'niqe', 'clipiqa', 'clipiqa+', 'dists'],
    'fastvqa': ['fastvqa', 'fastervqa'],
    'qalign': ['qalign'],
    'check': ['check']

}
FR_MODES = ['check', 'vmaf4k', 'vmaf', 'vmaf4k-full', 'vmaf-full', 'psnr', 'check', 'cvqa-fr', 'cvqa-fr-ms', 'lpips', 'dists']
NR_MODES = ['cvqa-nr', 'cvqa-nr-ms', 'dover', 'cover', 'uvq', 'maxvqa', 'musiq', 'qalign', 'fastvqa', 'fastervqa', 'brisque', 'niqe', 'clipiqa', 'clipiqa+']
AVAILABLE_MODES = [mode for sublist in MODES.values() for mode in sublist]


def check_model_availability(mode, rebuild=False):
    if mode in MODES['dover']:
        from metrics.dover import check_dover
        if not check_dover():
            return False

    if mode in MODES['cover']:
        from metrics.cover import check_cover
        if not check_cover():
            return False

    if mode in MODES['cvqa']:
        from metrics.cvqa import check_cvqa
        if not check_cvqa():
            return False

    if mode in MODES['uvq']:
        from metrics.uvq import check_uvq
        if not check_uvq():
            return False

    if mode in MODES['maxvqa']:
        from metrics.maxvqa import check_maxvqa
        if not check_maxvqa():
            return False

    if mode in MODES['pyiqa']:
        from metrics.pyiqa import check_pyiqa
        if not check_pyiqa(mode):
            return False
    
    if mode in MODES['fastvqa']:
        from metrics.fastvqa import check_fastvqa
        if not check_fastvqa():
            return False

    if mode in MODES['qalign']:
        from metrics.qalign import check_qalign
        if not check_qalign(rebuild=rebuild):
            return False

    return True


def compare_video_properties(reference, distorted):
    ref_info = get_video_info(reference)
    dist_info = get_video_info(distorted)
    
    if not ref_info or not dist_info:
        print_key_value("ERROR", "Could not retrieve video information", force=True)
        return False
    
    print_line()
    print_line(f"{'ATTRIBUTE':<13} {'REFERENCE':<14} {'DISTORTED':<14}")
    print_line(f"{'Resolution':<13} {ref_info['resolution']:<14} {dist_info['resolution']:<14}")
    print_line(f"{'Framerate':<13} {ref_info['fps']:.3f} fps{'':<4} {dist_info['fps']:.3f} fps{'':<4}")
    print_line(f"{'Frame count':<13} {ref_info['frame_count']:<14} {dist_info['frame_count']:<14}")
    print_line(f"{'Duration':<13} {format_duration(ref_info['duration']):<14} {format_duration(dist_info['duration']):<14}")
    print_line(f"{'Pixel format':<13} {ref_info['pix_fmt']:<14} {dist_info['pix_fmt']:<14}")
    print_line(f"{'Color range':<13} {ref_info['color_range']:<14} {dist_info['color_range']:<14}")
    print_line(f"{'File size':<13} {format_file_size(ref_info['file_size']):<14} {format_file_size(dist_info['file_size']):<14}")
        
    messages = [] 
    if ref_info['width'] != dist_info['width'] or ref_info['height'] != dist_info['height']:
        messages.append(("ERROR", "Resolution mismatch"))
    
    if ref_info['frame_count'] != dist_info['frame_count']:
        messages.append(("ERROR", "Frame count mismatch"))

    fps_tolerance = 0.001
    if abs(ref_info['fps'] - dist_info['fps']) > fps_tolerance:
        messages.append(("WARNING", "Framerate mismatch"))

    if ref_info['color_range'] != dist_info['color_range'] and ref_info['color_range'] != 'unknown' and dist_info['color_range'] != 'unknown':
        messages.append(("WARNING", "Color range mismatch"))
    
    if ref_info['pix_fmt'] != dist_info['pix_fmt']:
        messages.append(("WARNING", "Pixel format mismatch"))

    
    has_errors = any(level == "ERROR" for level, _ in messages)
    has_warnings = any(level == "WARNING" for level, _ in messages)
    if len(messages) > 0:
        print_line()

    for level, msg in messages:
        print_key_value(level, msg)
        
    if has_errors:
        return False
    
    if has_warnings:
        print_line("This may affect results but analysis will continue...")
        
    return True


def run_analysis(mode, distorted, reference=None, output_dir=None, verbose=True):
    properties_match = True
    scale = None
    if mode in FR_MODES and reference is not None:
        properties_match = compare_video_properties(reference, distorted)

        if mode == 'check':
            return properties_match, None
        
        elif not properties_match and 'vmaf' or 'psnr' in mode:
            ref_info = get_video_info(reference)
            dis_info = get_video_info(distorted)
            if ref_info is not None and dis_info is not None and (ref_info['width'] != dis_info['width'] or ref_info['height'] != dis_info['height']):
                scale = (ref_info['width'], ref_info['height'])
                print_line(f"Scaling distorted video to {scale[0]}x{scale[1]} for analysis", force=True)
            else: 
                return properties_match, None 

        elif not properties_match:
            return properties_match, None

    if mode in MODES['ffmpeg']:
        from metrics.ffmpeg import run_ffmpeg
        return properties_match, run_ffmpeg(mode, distorted, reference, scale, output_dir)
    elif mode in MODES['cvqa']:
        from metrics.cvqa import run_cvqa
        return properties_match, run_cvqa(mode, distorted, reference, output_dir)
    elif mode in MODES['lpips']:
        from metrics.lpips import run_lpips
        return properties_match, run_lpips(mode, distorted, reference, output_dir)
    elif mode in MODES['dover']:
        from metrics.dover import run_dover
        return properties_match, run_dover(mode, distorted, output_dir)
    elif mode in MODES['cover']:
        from metrics.cover import run_cover
        return properties_match, run_cover(mode, distorted, output_dir)
    elif mode in MODES['uvq']:
        from metrics.uvq import run_uvq
        return properties_match, run_uvq(mode, distorted, output_dir)
    elif mode in MODES['maxvqa']:
        from metrics.maxvqa import run_maxvqa
        return properties_match, run_maxvqa(mode, distorted, reference, output_dir)
    elif mode in MODES['pyiqa']:
        from metrics.pyiqa import run_pyiqa
        return properties_match, run_pyiqa(mode, distorted, reference, output_dir)
    elif mode in MODES['fastvqa']:
        from metrics.fastvqa import run_fastvqa
        return properties_match, run_fastvqa(mode, distorted, output_dir)
    elif mode in MODES['qalign']:
        from metrics.qalign import run_qalign
        return properties_match, run_qalign(mode, distorted, output_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    
    parser = argparse.ArgumentParser(description='Run video quality analysis comparing a distorted video against a reference video')
    parser.add_argument("-d", '--distorted', required=True, help='Distorted (compressed) video file or folder')
    parser.add_argument("-r", '--reference', help='Reference (original) video file or folder (required for FR methods)')
    parser.add_argument("-m", '--mode', choices=AVAILABLE_MODES, default='vmaf4k-full')
    parser.add_argument('-o', '--output', nargs='?', const='.', help='Save output files. Optional: specify directory (default: same as distorted file)')
    parser.add_argument('-q', '--quiet', default=False, action='store_true', help='Enable quiet output')
    parser.add_argument('--rebuild', default=False, action='store_true', help='Delete and rebuild the selected model if applicable')
    args = parser.parse_args()

    set_quiet_mode(args.quiet)

    print_separator("STARTING VQCHECK")

    if os.path.isfile(args.distorted):
        distorted_files = [args.distorted, ]
    else:
        distorted_files = get_video_files(args.distorted)
    
    reference_files = []
    if args.reference:
        if os.path.isfile(args.reference):
            reference_files = [args.reference,]
        else:
            reference_files = get_video_files(args.reference)

    print_key_value("Distorted", f"{args.distorted} ({len(distorted_files)})")
    if args.reference:
        print_key_value("Reference", f"{args.reference} ({len(reference_files)})")

    if args.mode in FR_MODES and not args.reference:
        print_line("ERROR: Reference video is required for the selected mode", force=True)
        return

    # just print
    if args.output:
        if args.output == '.':
            print_key_value("Output", "(same as distorted files)")
        else:
            print_key_value("Output", f"{args.output}")

    print_key_value("Mode", f"{args.mode}")
    if not check_model_availability(args.mode, args.rebuild):
        return
    
    if args.mode in NR_MODES and args.reference:
        args.reference = None # Ignore reference for NR modes

    total_files = len(distorted_files)
    matching_properties = 0
    perfect_match = 0
    
    for distorted in distorted_files:
        if not reference_files:
            reference = None
        if len(reference_files) == 1:
            reference = reference_files[0]
        elif len(reference_files) > 1:
            reference = find_reference_file(distorted, reference_files)

        if not reference and args.mode in FR_MODES:
            print_line(f"ERROR: No reference file found for {distorted}", force=True)
            continue
        
        output_dir = None
        if args.output is not None:
            if args.output == '.':
                output_dir = os.path.dirname(distorted)
            else:
                output_dir = args.output

        print_separator(f"VQCheck ({args.mode})", newline=True)
        print_key_value("Distorted", distorted, force=True)
        if reference:
            print_key_value("Reference", reference)
        properties_match, results = run_analysis(args.mode, distorted, reference, output_dir)
        if not properties_match:
            print_line("SKIPPED (property mismatch)", force=True)
        print_separator()

        if properties_match:
            matching_properties += 1
            
        if results:
            if 'psnr' in args.mode:
                if results.get('psnr_avg', 0) == float('inf'):
                    perfect_match += 1
            elif 'lpips' in args.mode:
                if results.get('metadata', {}).get('mean_distance', 1) == 0:
                    perfect_match += 1
            else:
                if results.get('psnr', 0) >= 60: # == inf for VMAF tool
                    perfect_match += 1
    
    if total_files > 1:
        print_separator("SUMMARY", newline=True)
        print_key_value("Files Processed", str(total_files))
        if args.mode in FR_MODES:
            print_key_value("Matching Properties", str(matching_properties))
            print_key_value("Perfect Matches", str(perfect_match))
        print_separator()
    

if __name__ == "__main__":
    main()
