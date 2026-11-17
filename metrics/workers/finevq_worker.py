"""FineVQ worker: loads the model once, then scores videos like main() in internvl/train/inference.py
(with the flags of shell/infer.sh), minus torchrun/deepspeed and training-only steps.

Usage (inside the FineVQ venv, cwd and PYTHONPATH = FineVQ repo): python finevq_worker.py <weights_dir>
"""
import json
import os
import sys
import tempfile

from _vq_protocol import serve

SEED = 42


def load():
    import torch
    from transformers import AutoTokenizer, set_seed
    from internvl.model.internvl_chat_infer import InternVLChatConfig, InternVLChatModel
    from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                          IMG_START_TOKEN, QUAD_END_TOKEN, QUAD_START_TOKEN, REF_END_TOKEN,
                                          REF_START_TOKEN)
    from internvl.train.inference import DataTrainingArguments

    weights = sys.argv[1]
    data_args = DataTrainingArguments(conv_style='internlm2-chat', force_image_size=448, down_sample_ratio=0.5,
                                      max_dynamic_patch=6, max_seq_length=4096, dynamic_image_size=True,
                                      use_thumbnail=True)
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(weights, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = weights
    tokenizer.model_max_length = data_args.max_seq_length
    num_new_tokens = tokenizer.add_tokens([IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN, QUAD_START_TOKEN,
                                           QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN, BOX_START_TOKEN,
                                           BOX_END_TOKEN], special_tokens=True)

    config = InternVLChatConfig.from_pretrained(weights)
    config.vision_config.drop_path_rate = 0.1
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'
    config.template = data_args.conv_style
    config.select_layer = -1
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = 'v2'
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    model = InternVLChatModel.from_pretrained(weights, torch_dtype=torch.bfloat16, config=config)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size, patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings[-num_new_tokens:] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.eval().cuda()
    return model, tokenizer, data_args


def score(state, request):
    import torch
    from torch.utils.data import DataLoader
    from transformers import set_seed
    from internvl.train.inference import build_datasets

    model, tokenizer, data_args = state
    video = request['video']
    set_seed(SEED)  # upstream seeds once per run, i.e. once per video

    with tempfile.TemporaryDirectory(prefix='finevq_') as tmp:
        video_list = os.path.join(tmp, 'videos.txt')
        with open(video_list, 'w') as f:
            f.write(os.path.basename(video) + '\n')
        data_args.meta_path = os.path.join(tmp, 'meta.json')
        with open(data_args.meta_path, 'w') as f:
            json.dump({"vqtools": {"root": os.path.dirname(video), "video_name_txt": video_list,
                                   "data_augment": False, "repeat_time": 1, "length": 1}}, f)

        dataset = build_datasets(data_args, tokenizer, None, model, group_by_length=True,
                                 dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
                                 min_dynamic_patch=data_args.min_dynamic_patch,
                                 max_dynamic_patch=data_args.max_dynamic_patch, normalize_type=data_args.normalize_type)
        item = next(iter(DataLoader(dataset, batch_size=1)))

    with torch.no_grad():
        output = model(pixel_values2=item["pixel_values2"][0].to(model.device, torch.bfloat16),
                       pixel_values=item["pixel_values"][0].to(model.device, torch.bfloat16),
                       input_ids=item["input_ids"].to(model.device),
                       attention_mask=item["attention_mask"].to(model.device),
                       image_flags=item["image_flags"][0].to(model.device),
                       labels=item["labels"].to(model.device))
    return {'score': output['score1'].item()}


if __name__ == '__main__':
    serve(load, score)
