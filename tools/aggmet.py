import os
import json
import argparse
from collections import defaultdict

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
        print(filename)
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
    
    # Track successfully processed metrics for missing metrics report
    successfully_processed_metrics = defaultdict(set)
    all_successful_metric_types = set()
    
    # Track statistics for general report
    existing_entries_count = len(consolidated_data)
    new_entries_count = 0
    updated_entries_count = 0
    total_metric_files_found = 0
    failed_loads = 0
    
    # Process each base name
    for base_name, metrics in metric_files.items():
        # Count total metric files found
        total_metric_files_found += len(metrics)
        
        # Initialize or get existing entry
        was_existing = base_name in consolidated_data
        if base_name not in consolidated_data:
            consolidated_data[base_name] = {
                'name': base_name,
                'source_name': base_name.split('_')[0]
            }
            new_entries_count += 1
        else:
            updated_entries_count += 1
        
        entry = consolidated_data[base_name]
        
        renamed_keys = {
            'float_ssim': 'ssim',
            'float_ms_ssim': 'ms-ssim',
            'ms_ssim': 'ms-ssim',
            'compressed-vqa-nr': 'cvqa-nr',
            'compressed-vqa-fr': 'cvqa-fr',
            'compressed-vqa-fr-ms': 'cvqa-fr-ms'
        }

        for old_key, new_key in renamed_keys.items():
            if old_key in entry:
                entry[new_key] = entry.pop(old_key)
         
        if 'vmaf' in metrics:
            vmaf = load_json_if_exists(metrics['vmaf'])
            if vmaf:
                entry.update({
                    'psnr': (6 * vmaf['pooled_metrics']['psnr_y']['mean'] 
                            + vmaf['pooled_metrics']['psnr_cb']['mean'] 
                            + vmaf['pooled_metrics']['psnr_cr']['mean']) / 8,
                    'psnr_y': vmaf['pooled_metrics']['psnr_y']['mean'],
                    'psnr_cb': vmaf['pooled_metrics']['psnr_cb']['mean'],
                    'psnr_cr': vmaf['pooled_metrics']['psnr_cr']['mean'],
                    'ssim': vmaf['pooled_metrics']['float_ssim']['mean'],
                    'ms-ssim': vmaf['pooled_metrics']['float_ms_ssim']['mean'],
                    'vmaf': vmaf['pooled_metrics']['vmaf']['mean'],
                    'vmaf-neg': vmaf['pooled_metrics']['vmaf_neg']['mean'],
                })
                successfully_processed_metrics[base_name].add('vmaf')
                all_successful_metric_types.add('vmaf')
            else:
                failed_loads += 1
        
        # Process other metrics
        metric_mappings = {
            'avqbitsh0f': ('avqbitsh0f', lambda x: x["per_sequence"]),
            'lpips': ('lpips', lambda x: x["metadata"]['mean_distance']),
            'dover': ('dover', lambda x: x["dover"]),
            'dover': ('dover', lambda x: x["overall_score"]),
            'fastvqa': ('fastvqa', lambda x: x['fastervqa_score']),
            'musiq': ('musiq', lambda x: x['mean_musiq']),
            'qalign': ('qalign', lambda x: x['qalign_score']),
            'NRCompressedVQA': ('cvqa-nr', lambda x: x['score']),
            'FRCompressedVQA': ('cvqa-fr', lambda x: x['score']),
            'FRCompressedVQAMS': ('cvqa-fr-ms', lambda x: x['score']),
            'cvqa-nr': ('cvqa-nr', lambda x: x['score']),
            'cvqa-fr': ('cvqa-fr', lambda x: x['score']),
            'cvqa-fr-ms': ('cvqa-fr-ms', lambda x: x['score']),
            'p12044': ('p12044', lambda x: x['score']),
        }
        
        for metric_key, (output_key, extractor) in metric_mappings.items():
            if metric_key in metrics:
                data = load_json_if_exists(metrics[metric_key])
                if data:
                    try:
                        entry[output_key] = extractor(data)
                        successfully_processed_metrics[base_name].add(metric_key)
                        all_successful_metric_types.add(metric_key)
                    except (KeyError, TypeError):
                        print(f"Warning: Could not extract {metric_key} for {base_name}")
                        failed_loads += 1
                else:
                    failed_loads += 1
    
    # Generate comprehensive report
    output_list = list(consolidated_data.values())
    
    print("\n" + "="*60)
    print("CONSOLIDATION REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"📁 Metrics Directory: {args.metrics_dir}")
    print(f"📄 Output File: {args.output_file}")
    print(f"📋 Base File: {'Yes (' + args.existing_json + ')' if args.existing_json else 'No'}")
    print()
    
    # File processing statistics
    print("📊 PROCESSING STATISTICS:")
    print(f"   • Video entries processed: {len(metric_files)}")
    print(f"   • Metric files found: {total_metric_files_found}")
    print(f"   • Successful loads: {total_metric_files_found - failed_loads}")
    print(f"   • Failed loads: {failed_loads}")
    print()
    
    # Entry statistics
    if args.existing_json:
        print("📝 ENTRY STATISTICS:")
        print(f"   • Existing entries (from base file): {existing_entries_count}")
        print(f"   • New entries added: {new_entries_count}")
        print(f"   • Existing entries updated: {updated_entries_count}")
        print(f"   • Total entries in output: {len(output_list)}")
        print()
    
    # Metrics overview
    if all_successful_metric_types:
        print("🎯 METRICS OVERVIEW:")
        print(f"   • Unique metric types found: {len(all_successful_metric_types)}")
        print(f"   • Available metrics: {', '.join(sorted(all_successful_metric_types))}")
        print()
        
        # Missing metrics analysis
        if len(metric_files) > 1:
            missing_count = 0
            for base_name in metric_files.keys():
                missing_metrics = all_successful_metric_types - successfully_processed_metrics[base_name]
                if missing_metrics:
                    missing_count += 1
            
            if missing_count > 0:
                print("⚠️  MISSING METRICS WARNINGS:")
                for base_name in sorted(metric_files.keys()):
                    missing_metrics = all_successful_metric_types - successfully_processed_metrics[base_name]
                    if missing_metrics:
                        missing_list = sorted(missing_metrics)
                        print(f"   • {base_name}: missing {', '.join(missing_list)}")
                print(f"   • Total files with missing metrics: {missing_count}/{len(metric_files)}")
            else:
                print("✅ METRICS CONSISTENCY:")
                print("   • All files have consistent metrics")
        print()
    
    print("="*60)
    
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(output_list, f, indent=2)
    
    print(f"Consolidated {len(output_list)} entries to {args.output_file}")

if __name__ == "__main__":
    main()
