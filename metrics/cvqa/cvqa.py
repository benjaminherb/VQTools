import os
import json
from pathlib import Path
import tempfile
from datetime import datetime

from metrics.cvqa.cvqa_fr import run_compressed_vqa_fr
from metrics.cvqa.cvqa_nr import run_compressed_vqa_nr
from metrics.utils import get_output_filename, print_key_value, ts, print_line

MODEL_LINKS = {
    "UGCVQA_FR_model.pth": "https://drive.google.com/file/d/1ohKNe_r0bXBg7qp4vQj0mDT3CwJPHVMM/view?usp=sharing",
    "UGCVQA_NR_model.pth": "https://drive.google.com/file/d/1K73padYMgq70zVWVVLIODs9SyIhdgqkT/view?usp=sharing"
}

def check_cvqa():
    missing_models = []
    for model_name in MODEL_LINKS.keys():
        model_path = os.path.join('models/cvqa/ckpts', model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        print_line(f"Missing models: {', '.join(missing_models)}", force=True)
        print_line("Please download them manually from the provided links and add them to the 'models' directory.", force=True)
        for model_name in missing_models:
            print_line(f"{model_name}: {MODEL_LINKS[model_name]}", force=True)
        return False
    return True


def run_cvqa(mode, distorted, reference, output_dir=None):
    is_reference_based = 'cvqa-fr' in mode
    is_multiscale = 'ms' in mode
    
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
        temp_fd, output_file = tempfile.mkstemp(suffix='.json', prefix='cvqa_')
        os.close(temp_fd)
    
    start_time = datetime.now()
    
    try:
        print_line("\nRESULTS")
        print_key_value("Start Time", ts(start_time))


        if is_reference_based:
            run_compressed_vqa_fr(reference, distorted, output_file, multiscale=is_multiscale)
        else:
            run_compressed_vqa_nr(distorted, output_file, multiscale=is_multiscale)
        
        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")

        results = None
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
                score = results.get('score', 0)
                print_key_value(f"Score", f"{score:.4f}", force=True)

        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        
        return results

    except Exception as e:
        print_line(f"Error running CVQA analysis: {e}", force=True)
        if output_dir is None and os.path.exists(output_file):
            os.unlink(output_file)
        return None
