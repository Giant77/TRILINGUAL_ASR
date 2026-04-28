# Run via windows/powershell:
# overwrite : python main.py 2>&1 | Tee-Object -FilePath log/preprocessing_log.txt
# append    : python main.py 2>&1 | Tee-Object -FilePath log/preprocessing_log.txt -Append

# Run via linux/bash:
# overwrite : python main.py 2>&1 | tee log/preprocessing_log.txt
# append    : python main.py 2>&1 | tee -a log/preprocessing_log.txt

import argparse
import time
import os
import json
from preprocessing import *
from preprocess_audio import *

def parse_args():
    """Parse command line arguments for stage control."""
    parser = argparse.ArgumentParser(description="Trilingual ASR Preprocessing Pipeline")
    parser.add_argument('--start-stage', type=int, default=1, 
                        help='Start stage (default: 1)')
    parser.add_argument('--end-stage', type=int, default=1000, 
                        help='End stage (default: 1000)')
    return parser.parse_args()

def main():
    """
    Main pipeline with stage control (ESPnet-style):
      Stage 1: Audio Preprocessing - Process all audio, save to processed/
      Stage 2: Manifest Writes/Logs - Create manifests in manifests/original/
      Stage 3: Transcript Text Preprocessing - Process transcripts by language
      Stage 4: Balancing/Re-splitting - Balance data, save to manifests/balanced/
    """
    
    args = parse_args()
    start_stage = args.start_stage
    end_stage = args.end_stage
    start_time = time.time()
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE START")
    print(f"Stages: {start_stage} -> {end_stage}")
    print("="*70)
    
    # Registry of dataset processing functions (in order)
    processors = [
        # Indonesian
        ('id', process_id_cv),
        ('id', process_id_fleurs),
        # ('id', process_id_librivox), # skip: too much work for cleaning ejaan lama
        ('id', process_id_titml),
        ('id', process_id_indocsc),
        ('id', process_id_sindodsc),

        # Arabic
        ('ar', process_ar_cv),
        ('ar', process_ar_fleurs),
        ('ar', process_ar_clartts),

        # English
        ('en', process_en_librispeech),
        ('en', process_en_fleurs),
        ('en', process_en_cv_spon),

        # Code-Switching
        ('cs', process_cs_escwa),
        ('cs', process_cs_hari),
        ('cs', process_cs_homostoria),
    ]
    
    # Setup directories
    original_dir = os.path.join(BASE_OUT, "manifests", "original")
    balanced_dir = os.path.join(BASE_OUT, "manifests", "balanced")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(balanced_dir, exist_ok=True)
    
    lang_datasets = {}
    
    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Audio Preprocessing
    # ─────────────────────────────────────────────────────────────
    if start_stage <= 1 <= end_stage:
        print("\n" + "="*70)
        print("STAGE 1: Audio Preprocessing")
        print("="*70)
        
        for lang, processor_func in processors:
            processor_func(mode='audio')
    
    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Manifest Writes/Logs
    # ─────────────────────────────────────────────────────────────
    if start_stage <= 2 <= end_stage:
        print("\n" + "="*70)
        print("STAGE 2: Manifest Writes/Logs")
        print("="*70)
        
        for lang, processor_func in processors:
            result = processor_func(mode='manifest', manifest_dir=original_dir)
            dataset_key = processor_func.__name__.replace('process_', '')
            lang_datasets[dataset_key] = result
        
        # Save short-segment manifests
        print("\nSaving short-segment manifests...")
        save_short_segment_manifests(original_dir)
    
    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Transcript Text Preprocessing
    # ─────────────────────────────────────────────────────────────
    if start_stage <= 3 <= end_stage:
        print("\n" + "="*70)
        print("STAGE 3: Transcript Text Preprocessing")
        print("="*70)
        
        for dataset_key in sorted(lang_datasets.keys()):
            manifest_path = os.path.join(original_dir, f"{dataset_key}_manifest.json")
            if not os.path.exists(manifest_path):
                print(f"  Skipping {dataset_key}: manifest not found at {manifest_path}")
                continue
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            lang = dataset_key.split('_')[0]
            print(f"  Processing transcripts for {dataset_key} (lang={lang})")
            
            for split in ['train', 'dev', 'test']:
                if split in manifest and manifest[split]:
                    preprocess_transcripts(lang, manifest[split], dataset_key)
            
            # Save back to manifest
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Balancing/Re-splitting
    # ─────────────────────────────────────────────────────────────
    if start_stage <= 4 <= end_stage:
        print("\n" + "="*70)
        print("STAGE 4: Balancing/Re-splitting")
        print("="*70)
        
        balanced_records = balance_lang_data(lang_datasets)
        
        for dataset_key, original_result in lang_datasets.items():
            lang = dataset_key.split('_')[0]
            is_predetermined = original_result.get('is_predetermined', False)
            split_source = 'predetermined_full' if is_predetermined else '8_1_1'
            
            # Get balanced records (may be fewer after balancing)
            balanced_recs = balanced_records.get(dataset_key, original_result.get('all_records', []))
            
            # Re-split balanced data
            split_records = split_data(balanced_recs)
            
            # Generate dynamic split_note with detailed statistics
            orig_count = len(original_result.get('all_records', []))
            new_count = len(balanced_recs)
            orig_hours = calculate_hours(original_result.get('all_records', []))
            new_hours = calculate_hours(balanced_recs)
            train_h = round(calculate_hours(split_records.get('train', [])), 2)
            dev_h = round(calculate_hours(split_records.get('dev', [])), 2)
            test_h = round(calculate_hours(split_records.get('test', [])), 2)
            
            split_note = f"Balanced: {orig_count}→{new_count} utts, {orig_hours:.1f}→{new_hours:.1f}h, train={train_h}h dev={dev_h}h test={test_h}h"
            
            save_manifest(dataset_key, lang, split_source, split_note, split_records, manifest_dir=balanced_dir)
    
    # ─────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────
    end_time = time.time()
    runtime_sec = end_time - start_time
    runtime_min = runtime_sec / 60.0
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*70)
    print(f"Original manifests: {original_dir}")
    print(f"Balanced manifests: {balanced_dir}")
    print(f"Total runtime: {runtime_sec:.2f} seconds ({runtime_min:.2f} minutes)")
    print("Next: run local/manifests_to_kaldi.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
