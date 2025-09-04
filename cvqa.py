import os
import argparse
import subprocess
import json
import glob
from pathlib import Path
from models.cvqa.cvqa_fr import run_compressed_vqa_fr
from models.cvqa.cvqa_nr import run_compressed_vqa_nr

MODEL_LINKS = {
    "UGCVQA_FR_model.pth": "https://drive.google.com/file/d/1ohKNe_r0bXBg7qp4vQj0mDT3CwJPHVMM/view?usp=sharing",
    "UGCVQA_NR_model.pth": "https://drive.google.com/file/d/1K73padYMgq70zVWVVLIODs9SyIhdgqkT/view?usp=sharing"
}

def check_models():
    missing_models = []
    for model_name in MODEL_LINKS.keys():
        model_path = os.path.join('models/cvqa/ckpts', model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        print(f"Missing models: {', '.join(missing_models)}")
        print("Please download them manually from the provided links and add them to the 'models' directory.")
        return False
    return True

def run_cvqa_fr(reference, distorted, output_file, multiscale=False):
    try:
        run_compressed_vqa_fr(reference, distorted, output_file, multiscale)
    except Exception as e:
        print(f"Error: {e}")

def run_cvqa_nr(distorted, output_file, multiscale=False):
    try:
        run_compressed_vqa_nr(distorted, output_file, multiscale)
    except Exception as e:
        print(f"Error: {e}")

def find_matching_reference(distorted_path, reference_dir):
    """Find matching reference video for distorted video"""
    distorted_name = Path(distorted_path).stem
    # Remove common suffixes like quality indicators
    base_name = distorted_name.split('_')[0]
    
    for ref_pattern in [f"{base_name}*", f"{distorted_name.split('_')[0]}*"]:
        matches = glob.glob(os.path.join(reference_dir, ref_pattern))
        if matches:
            return matches[0]
    return None

def main():
    if not check_models():
        return

    parser = argparse.ArgumentParser(description="Run CVQA on videos")
    parser.add_argument("-d", "--distorted", required=True, help="Distorted video file or directory")
    parser.add_argument("-r", "--reference", help="Reference video file or directory (for FR model)")
    parser.add_argument("-o", "--output_dir", default="./results", help="Output directory")
    parser.add_argument("-m", "--model", choices=['fr', 'nr'], required=True, help="Model type")
    parser.add_argument("--multiscale", action="store_true", help="Use multiscale method")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of distorted videos
    if os.path.isfile(args.distorted):
        distorted_videos = [args.distorted]
    else:
        distorted_videos = glob.glob(os.path.join(args.distorted, "*.mkv"))

    for distorted in distorted_videos:
        video_name = Path(distorted).stem
        output_file = os.path.join(args.output_dir, f"{video_name}.cvqa-{args.model}{'-ms' if args.multiscale else ''}.json")

        if os.path.exists(output_file):
            print(f"Skipping {video_name} (already processed)")
            continue

        print(f"Processing {video_name}...")
        
        if args.model == 'fr':
            if not args.reference:
                print("Error: Reference required for FR model")
                continue
                
            if os.path.isfile(args.reference):
                reference = args.reference
            else:
                reference = find_matching_reference(distorted, args.reference)
                if not reference:
                    print(f"No matching reference found for {video_name}")
                    continue

            result = run_cvqa_fr(reference, distorted, output_file, args.multiscale)
        else:
            result = run_cvqa_nr(distorted, output_file, args.multiscale)

        if not result:
            print(f"Error processing {video_name}")
        else:
            print(f"Completed {video_name}")

if __name__ == "__main__":
    main()