"""Q-Align worker: loads the video scorer once (same as `q_align/evaluate/scorer.py --video`).

Usage (inside the Q-Align venv): python qalign_worker.py <device> <model_path>
"""
import sys

from _vq_protocol import serve


def load():
    from q_align.evaluate.scorer import QAlignVideoScorer, load_video
    device, model_path = sys.argv[1:3]
    return QAlignVideoScorer(pretrained=model_path, device=device), load_video


def score(state, request):
    scorer, load_video = state
    return {'score': scorer([load_video(request['video'])]).tolist()[0]}


if __name__ == '__main__':
    serve(load, score)
