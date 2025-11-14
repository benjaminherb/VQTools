import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

metric_configs = {
    # metric name (*.name.json), output key in final json, extractor functions to get the value from loaded json
    'vmaf':[('vmaf', lambda x: x['pooled_metrics']['vmaf']['mean']),
            ('vmaf', lambda x: x['pooled_metrics']['vmaf_4k']['mean']),
            ('vmaf-neg', lambda x: x['pooled_metrics']['vmaf_neg']['mean']),
            ('vif-scale0', lambda x: x['pooled_metrics']['integer_vif_scale0']['mean']),
            ('vif-scale0-egl', lambda x: x['pooled_metrics']['integer_vif_scale0_egl_1']['mean']),
            ('vif-scale1', lambda x: x['pooled_metrics']['integer_vif_scale1']['mean']),
            ('vif-scale1-egl', lambda x: x['pooled_metrics']['integer_vif_scale1_egl_1']['mean']),
            ('vif-scale2', lambda x: x['pooled_metrics']['integer_vif_scale2']['mean']),
            ('vif-scale2-egl', lambda x: x['pooled_metrics']['integer_vif_scale2_egl_1']['mean']),
            ('vif-scale3', lambda x: x['pooled_metrics']['integer_vif_scale3']['mean']),
            ('vif-scale3-egl', lambda x: x['pooled_metrics']['integer_vif_scale3_egl_1']['mean']),
            ('adm2', lambda x: x['pooled_metrics']['integer_adm2']['mean']),
            ('adm-scale0', lambda x: x['pooled_metrics']['integer_adm_scale0']['mean']),
            ('adm-scale1', lambda x: x['pooled_metrics']['integer_adm_scale1']['mean']),
            ('adm-scale2', lambda x: x['pooled_metrics']['integer_adm_scale2']['mean']),
            ('adm-scale3', lambda x: x['pooled_metrics']['integer_adm_scale3']['mean']),
            ('motion', lambda x: x['pooled_metrics']['integer_motion']['mean']),
            ('motion2', lambda x: x['pooled_metrics']['integer_motion2']['mean']),
            ('psnr', lambda x: (6 * x['pooled_metrics']['psnr_y']['mean'] 
                                  + x['pooled_metrics']['psnr_cb']['mean'] 
                                  + x['pooled_metrics']['psnr_cr']['mean']) / 8),
            ('ssim', lambda x: x['pooled_metrics']['float_ssim']['mean']),
            ('ms-ssim', lambda x: x['pooled_metrics']['float_ms_ssim']['mean']),
            ('psnr-y', lambda x: x['pooled_metrics']['psnr_y']['mean']),
            ('psnr-cb', lambda x: x['pooled_metrics']['psnr_cb']['mean']),
            ('psnr-cr', lambda x: x['pooled_metrics']['psnr_cr']['mean']),
            ('psnr-hvs', lambda x: x['pooled_metrics']['psnr_hvs']['mean'])],
    'avqbitsm0': [('avqbitsm0', lambda x: x["per_sequence"])],
    'avqbitsm1': [('avqbitsm1', lambda x: x["per_sequence"])],
    'avqbitsh0f': [('avqbitsh0f', lambda x: x["per_sequence"])],
    'lpips': [('lpips', lambda x: x["metadata"]['mean_distance'])],
    'dover': [('dover', lambda x: x["dover"]),
              ('dover_aesthetic', lambda x: x["cover_res_0"]),
              ('dover_technical', lambda x: x["cover_res_1"]),
              ('dover', lambda x: x["fused_score"]),
              ('dover_aesthetic', lambda x: x["aesthetic_score"]),
              ('dover_technical', lambda x: x["technical_score"])],
    'cover': [('cover', lambda x: x['fused_score']),
              ('cover_aesthetic', lambda x: x['aesthetic_score']),
              ('cover_technical', lambda x: x['technical_score']),
              ('cover_semantic', lambda x: x['semantic_score'])],
    'maxvqa': [('maxvqa', lambda x: x['overall_score']),
               ('maxvqa_quality', lambda x: x['high quality vs low quality']),
               ('maxvqa_content', lambda x: x['good content vs bad content']),
               ('maxvqa_composition', lambda x: x['organized composition vs chaotic composition']),
               ('maxvqa_color', lambda x: x['vibrant color vs faded color']),
               ('maxvqa_lighting', lambda x: x['contrastive lighting vs gloomy lighting']),
               ('maxvqa_trajectory', lambda x: x['consistent trajectory vs incoherent trajectory']),
               ('maxvqa_aesthetics', lambda x: x['good aesthetics vs bad aesthetics']),
               ('maxvqa_sharpness', lambda x: x['sharp vs fuzzy']),
               ('maxvqa_focus', lambda x: x['in-focus vs out-of-focus']),
               ('maxvqa_noise', lambda x: x['noiseless vs noisy']),
               ('maxvqa_motion', lambda x: x['clear-motion vs blurry-motion']),
               ('maxvqa_stability', lambda x: x['stable vs shaky']),
               ('maxvqa_exposure', lambda x: x['well-exposed vs poorly-exposed']),
               ('maxvqa_compression', lambda x: x['original vs compressed']),
               ('maxvqa_fluency', lambda x: x['fluent vs choppy']),
               ('maxvqa_clarity', lambda x: x['clear vs severely degraded'])],
    'uvq': [('uvq', lambda x: x['uvq']),
            ('uvq_compression', lambda x: x['compression']),
            ('uvq_content', lambda x: x['content']),
            ('uvq_distortion', lambda x: x['distortion']),
            ('uvq_compression_content', lambda x: x['compression_content']),
            ('uvq_compression_distortion', lambda x: x['compression_distortion']),
            ('uvq_content_distortion', lambda x: x['content_distortion'])],
    'fastvqa': [('fastvqa', lambda x: x['score'])],
    'fastervqa': [('fastervqa', lambda x: x['fastervqa_score']),
                  ('fastervqa', lambda x: x['score'])],
    'brisque': [('brisque', lambda x: x['mean_score'])],
    'niqe': [('niqe', lambda x: x['mean_score'])],
    'clipiqa': [('clipiqa', lambda x: x['mean_score'])],
    'clipiqa+': [('clipiqa+', lambda x: x['mean_score'])],
    'dists': [('dists', lambda x: x['mean_score'])],
    'musiq': [('musiq', lambda x: x['mean_musiq']),
              ('musiq', lambda x: x['mean_score'])],
    'vila': [('vila', lambda x: x['mean_vila'])],
    'qalign': [('qalign', lambda x: x['qalign_score']),
               ('qalign', lambda x: x['score'])],
    'cvqa-nr': [('cvqa-nr', lambda x: x['score'])],
    'cvqa-nr-ms': [('cvqa-nr-ms', lambda x: x['score'])],
    'cvqa-fr': [('cvqa-fr', lambda x: x['score'])],
    'cvqa-fr-ms': [('cvqa-fr-ms', lambda x: x['score'])],
    'p12043': [('p12043', lambda x: x['per_sequence'])],
    'p12044': [('p12044', lambda x: x['score'])],
    'siti': [('si', lambda x: x['aggregated_statistics']['si']['mean']),
             ('ti', lambda x: x['aggregated_statistics']['ti']['mean'])],
    'clip': [('clip', lambda x: x['clip'][0])]
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Consolidate video quality metrics into a single JSON file')
    parser.add_argument('--metrics-dir', '-m', required=True, type=str, 
                       help='Directory containing individual metric JSON files')
    parser.add_argument('--output-file', '-o', required=True, type=str,
                       help='Output JSON file path for consolidated metrics')
    parser.add_argument('--existing-json', '-e', type=str, default=None,
                       help='Path to existing combined JSON file to update (optional)')
    return parser.parse_args()

def load_json_if_exists(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return None

def extract_name_and_metric(filename):
    """Extract base name from metric filename (e.g., 'video1.vmaf.json' -> 'video1')"""
    parts = filename.split('.')
    if len(parts) >= 3 and parts[-1] == 'json':
        base_name = '.'.join(parts[:-2])
        metric_type = parts[-2]  # e.g., 'vmaf', 'dover', etc.
        return base_name, metric_type

    return None, None

def main():
    args = parse_arguments()
    
    # Load existing data if provided
    consolidated_data = {}
    if args.existing_json:
        existing_data = load_json_if_exists(args.existing_json)
        if existing_data:
            if isinstance(existing_data, list):
                consolidated_data = {item.get('name', f'item_{i}'): item for i, item in enumerate(existing_data)}
            else:
                consolidated_data = existing_data
    
    metric_files = defaultdict(dict)
    
    for root, dirs, files in os.walk(args.metrics_dir):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            
            base_name, metric_type = extract_name_and_metric(filename)
            if not base_name:
                continue
            
            metric_files[base_name][metric_type] = os.path.join(root, filename)
    
    errors = {
        'failed_load': [],
        'failed_extract': {
            metric_key: {output_key: [] for output_key, _ in config}
            for metric_key, config in metric_configs.items()
        },
    }
    
    for base_name, metrics in tqdm(metric_files.items(), desc="Parsing Metrics", unit="video"):
        if base_name not in consolidated_data:
            consolidated_data[base_name] = {
                'name': base_name,
            }

        if 'meta' in metrics:
            metadata = load_json_if_exists(metrics['meta'])
            for key, value in metadata.items():
                if key in ('bitrate', 'bit_rate',  'bpp', 'bit_depth', 'fps', 'frame_rate', 'framerate', 'frames', 'filesize', 'encoding_time', 'encoding_speed'):
                    consolidated_data[base_name][key] = value
                if key == 'upscaling':
                    print('upscaling metadata:', value)
                    consolidated_data[base_name]['upscaling_spf'] = value['spf']

        entry = consolidated_data[base_name]
         
        for metric_key, config in metric_configs.items():
            if not metric_key in metrics:
                continue

            data = load_json_if_exists(metrics[metric_key])
            if not data:
                errors['failed_load'].append(metrics[metric_key])
                continue

            for config_item in config:
                output_key, extractor = config_item
                try:
                    if output_key in entry and entry[output_key] is not None:
                        continue  # Skip if already exists
                    extracted_value = extractor(data)
                    entry[output_key] = extracted_value
                except (KeyError, TypeError):
                    errors['failed_extract'][metric_key][output_key].append(metrics[metric_key])

    
    output_list = list(consolidated_data.values())
    
    total_metric_files = sum(len(metrics) for metrics in metric_files.values())
    
    all_metric_keys = set()
    metric_coverage = defaultdict(int)
    
    for entry in output_list:
        for key in entry.keys():
            if key not in ['name']:  # Skip non-metric keys
                all_metric_keys.add(key)
                metric_coverage[key] += 1
    
    total_entries = len(output_list)
    fully_covered_metrics = [k for k, v in metric_coverage.items() if v == total_entries]
    partially_covered_metrics = [k for k, v in metric_coverage.items() if 0 < v < total_entries]
    missing_metrics = [k for k in all_metric_keys if metric_coverage[k] == 0]
    
    print("\n" + "="*60)
    print("CONSOLIDATION REPORT")
    print("="*60)
    
    print(f"Metrics Directory: {args.metrics_dir}")
    print(f"Output File: {args.output_file}")
    print(f"Base File: {'Yes (' + args.existing_json + ')' if args.existing_json else 'No'}")
    print()
    
    print("FILES:")
    print(f"   • Video entries processed: {len(metric_files)}")
    print(f"   • Metric files found: {total_metric_files}")
    print()
    
    if all_metric_keys:
        print("METRICS:")
        print(f"   • Total unique metrics found: {len(all_metric_keys)}")
        print(f"   • Fully covered metrics ({len(fully_covered_metrics)}): {', '.join(sorted(fully_covered_metrics))}")
        if partially_covered_metrics:
            partial_info = [f"{k} ({metric_coverage[k]}/{total_entries})" for k in sorted(partially_covered_metrics)]
            print(f"   • Partially covered metrics: {', '.join(partial_info)}")
        if missing_metrics:
            print(f"   • Missing metrics: {', '.join(sorted(missing_metrics))}")
        print()

    failed_output_keys = []
    for metric_key, output_dict in errors['failed_extract'].items():
        for output_key, file_list in output_dict.items():
            if file_list and output_key not in fully_covered_metrics:
                failed_output_keys.append(f"{output_key} ({len(file_list)})")


    if errors['failed_load'] or failed_output_keys:
        print("ERRORS:")
    if errors['failed_load']:
        print(f"   • Failed to load JSON files ({len(errors['failed_load'])})")
        for filepath in errors['failed_load']:
            print(f"       - {filepath}")
    if failed_output_keys:
        print(f"   • Failed to extract metrics: {', '.join(sorted(failed_output_keys))}")
    print()

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(output_list, f, indent=2)
    
    print("="*60)    
    print(f"Consolidated {len(output_list)} entries to {args.output_file}")

if __name__ == "__main__":
    main()