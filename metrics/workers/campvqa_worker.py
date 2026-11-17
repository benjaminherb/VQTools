"""CAMP-VQA worker: loads all models once, then runs the per-video part of src/camp-vqa_demo.py.

Usage (inside the CAMP-VQA venv, cwd = src/): python campvqa_worker.py <camp-vqa_demo.py flags>
"""
import importlib.util
import os
import sys

from _vq_protocol import serve


def load():
    sys.path.insert(0, os.getcwd())
    # camp-vqa_demo.py cannot be imported by name (dash in the file name)
    spec = importlib.util.spec_from_file_location('camp_vqa_demo', 'camp-vqa_demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    # same as the demo's __main__, without the video-specific part
    config = demo.parse_arguments()
    device = demo.setup_device(config)
    models = (
        demo.SlowFast().to(device),
        demo.SwinT(model_name='swin_large_patch4_window7_224', global_pool='avg', pretrained=True).to(device),
        *demo.clip.load("ViT-B/32", device=device),
        demo.Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True),
        demo.Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device),
        demo.load_prompts(config.prompt_path),
    )
    if config.intra_cross_experiment == 'intra' and config.train_data_name != 'lsvq_train':
        from model_regression import Mlp, preprocess_data
    else:
        from model_regression_lsvq import Mlp, preprocess_data
    demo.preprocess_data = preprocess_data  # evaluate_video_quality() expects it as a module global
    model_mlp = demo.load_model(config, device, Mlp, 13056)
    return demo, config, device, models, model_mlp


def score(state, request):
    demo, config, device, models, model_mlp = state
    video_path = request['video']

    width, height, bitrate, bitdepth, framerate = demo.get_video_metadata(video_path)
    test_df = demo.pd.DataFrame({
        'vid': [os.path.splitext(os.path.basename(video_path))[0]],
        'test_data_name': [config.test_data_name],
        'test_video_path': [video_path],
        'prediction_mode': [config.prediction_mode],
        'width': [width], 'height': [height], 'bitrate': [bitrate], 'bitdepth': [bitdepth], 'framerate': [framerate]})
    top_n = int(config.target_size / config.patch_size) * int(config.target_size / config.patch_size)
    dataset = demo.VideoDataset_feature(test_df, os.path.dirname(video_path), config.test_data_name,
                                        demo.get_transform(config.resize), config.resize, config.patch_size,
                                        config.target_size, top_n)
    data_loader = demo.torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers,
                                                   pin_memory=device.type == "cuda")
    return {'score': demo.evaluate_video_quality(data_loader, *models, model_mlp, device)}


if __name__ == '__main__':
    serve(load, score)
