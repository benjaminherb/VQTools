import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from metrics.cover import MODEL_FILES
from metrics.utils import get_output_filename, save_json, print_key_value, ts, check_docker, build_docker_image, print_line, print_separator, modify_file, create_venv, run_in_venv
from metrics.dover import check_dover


def check_maxvqa():
    """Check if MAXVQA is available."""

    repo = Path(__file__).parent / "maxvqa"
    if not repo.exists():
        print_separator("BUILDING MAXVQA", newline=True)
        print_line("Cloning MaxVQA repository...", force=True)
        try:
            subprocess.run(['git', 'clone', 'https://github.com/VQAssessment/ExplainableVQA.git', str(repo)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            modify_file(str(repo / 'demo_maxvqa.py'), [
                {'action': 'replace', 'pattern': '"cuda"', 'content': '"cpu"'},
                {'action': 'replace', 'pattern': ".cuda()", 'content': '.to(device)'},
                {'action': 'replace', 'pattern': 'fast_vqa_encoder.load_state_dict(torch.load("../DOVER/pretrained_weights/DOVER.pth"),strict=False)', 'content': 'fast_vqa_encoder.load_state_dict(torch.load("../DOVER/pretrained_weights/DOVER.pth", map_location=torch.device(device)),strict=False)'},
                {'action': 'replace', 'pattern': 'state_dict = torch.load("maxvqa_maxwell.pt")', 'content': 'state_dict = torch.load("maxvqa_maxwell.pt", map_location=torch.device(device))'},
                {'action': 'delete', 'from': 158, 'to': -1},
                {'action': 'replace', 'pattern': 'import gradio as gr', 'content': '# import gradio as gr'},
                {'action': 'insert', 'line': 157, 'content': [
                    "        output = list(res.cpu().numpy()[0])",
                    "        results = {}",
                    "        for i, (pos, neg, score) in enumerate(zip(positive_descs, negative_descs, output)):",
                    "            dim_name = f'{pos} vs {neg}'",
                    "            results[dim_name] = float(rescale(score, i))",
                    "        results['overall_score'] = sum(results.values()) / len(results)",
                    "        print('MAXVQA_RESULTS_START')",
                    "        print(results)",
                    "        print('MAXVQA_RESULTS_END')",
                    "if __name__ == '__main__':",
                    "   import sys",
                    "   if len(sys.argv) > 1:",
                    "       video_path = sys.argv[1]",
                    "       inference(video_path)"
                ]}
            ])
            modify_file(str(repo / 'model' / 'maxvqa.py'), [
                {'action': 'replace', 'pattern': 'n_prompts.cuda()', 'content': 'n_prompts.cpu()'},
                {'action': 'replace', 'pattern': 'self.device = "cuda"', 'content': 'self.device = "cpu"'},
                {'action': 'replace', 'pattern': '.cuda()', 'content': '.cpu()'},
                {'action': 'replace', 'pattern': 'x = x.permute(1, 0, 2)  # NLD -> LND', 'content': '# x = x.permute(1, 0, 2) # NLD -> LND'},
                {'action': 'replace', 'pattern': 'x = x.permute(1, 0, 2)  # LND -> NLD', 'content': '# x = x.permute(1, 0, 2) # LND -> NLD'},
            ])

        except subprocess.CalledProcessError as e:
            print_line(f"ERROR: Failed to clone MaxVQA repository: {e}", force=True)
            return False

        if not (repo / 'venv').exists():
            print_line("Creating MAXVQA virtual environment...")
            create_venv(str(repo / 'venv'), python='python3.10', compile_decord=True)
            run_in_venv(str(repo / 'venv'), ['pip', 'install', 'pyyaml', 'scipy', 'scikit-learn', 'numpy==1.24.3', 'opencv-python'])
            subprocess.run(['git', 'clone', 'https://github.com/mlfoundations/open_clip.git', str(repo / 'open_clip')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            modify_file(str(repo / 'open_clip' / 'src' / 'open_clip' / 'modified_resnet.py'), [
                {'action': 'replace', 'pattern': 'return x[0]', 'content': 'return x'}
            ])
            run_in_venv(str(repo / 'venv'), ['pip', 'install', '-e', str(repo / 'open_clip')])
            check_dover()
            run_in_venv(str(repo / 'venv'), ['pip', 'install', '-e', str(repo.parent / 'dover')], work_dir=str(repo.parent / 'dover'))


        return True

    return True


def run_maxvqa(mode, distorted, reference, output_dir=None):
    """Run MAXVQA video quality assessment."""

    # Prepare output file
    output_file = None
    if output_dir is not None:
        output_file = get_output_filename(distorted, mode, output_dir)
        if os.path.exists(output_file):
            print_line(f"{output_file} exists already - SKIPPING!", force=True)
            return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = datetime.now()
    print_line("\nRESULTS")
    print_key_value("Start Time", ts(start_time))

    try:
        cmd = ['python', 'demo_maxvqa.py', os.path.abspath(distorted)]

        repo = Path(__file__).parent / "maxvqa"
        result = run_in_venv(str(repo / 'venv'), cmd, work_dir=str(repo))

        if result.returncode != 0:
            print_line(f"ERROR: MAXVQA evaluation failed: {result.stderr}", force=True)
            return None

        results = _parse_maxvqa_results(result.stdout, result.stderr, distorted)

        end_time = datetime.now()
        analysis_duration = end_time - start_time

        print_key_value("End Time", ts(end_time))
        print_key_value("Duration", f"{analysis_duration.total_seconds():.2f}s")

        if results:
            print_key_value("Overall Score", f"{results['overall_score']:.4f}", force=True)
            for key, value in results.items():
                if key not in ['timestamp', 'distorted', 'overall_score']:
                    print_key_value(f"{key.title()}", f"{value:.4f}")

        if output_file:
            save_json(results, output_file)

        return results

    except Exception as e:
        print_line(f"ERROR: MAXVQA evaluation failed: {e}", force=True)
        return None


def _parse_maxvqa_results(stdout, stderr, video_path=None):
    """Parse MAXVQA results from demo_maxvqa output."""
    import ast
    
    try:
        start_marker = 'MAXVQA_RESULTS_START'
        end_marker = 'MAXVQA_RESULTS_END'
        
        start_idx = stdout.find(start_marker) + len(start_marker)
        end_idx = stdout.find(end_marker)
        results_section = stdout[start_idx:end_idx].strip()
        
        results_dict = ast.literal_eval(results_section)
        
        results = {
            'timestamp': ts(),
            'distorted': os.path.basename(video_path) if video_path else '',
        }
        
        for key, value in results_dict.items():
            results[key] = value
        
        return results
            
    except Exception as e:
        print_line(f"ERROR: Could not parse MaxVQA results: {e}", force=True)
        return None
