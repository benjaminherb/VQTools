"""JSON-lines server for persistent model workers (client: ModelWorker in metrics/utils.py).

Runs inside the model venvs, so it only uses the standard library (python >= 3.9).
"""
import json
import os
import sys
import traceback


def serve(load_fn, score_fn):
    """Load the model once, then answer one JSON request per stdin line until stdin is closed."""
    # keep the real stdout for replies; all other output (prints, tqdm, C libraries) goes to stderr
    out = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)

    state = load_fn()
    out.write('{"ready": true}\n')
    for line in sys.stdin:
        try:
            reply = {'result': score_fn(state, json.loads(line))}
        except Exception:
            reply = {'error': traceback.format_exc()}
        out.write(json.dumps(reply) + '\n')
