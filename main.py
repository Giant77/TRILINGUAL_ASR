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
    parser.add_argument('--start-stage', '--stage', type=int, default=1, 
                        help='Start stage (default: 1)')
    parser.add_argument('--end-stage', '--stop-stage', type=int, default=1000, 
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
        print("\n" + "=" * 70)
        print("STAGE 3: Transcript Text Preprocessing")
        print("=" * 70)

        for lang, processor_func in processors:

            dataset_key = processor_func.__name__.replace(
                "process_",
                ""
            )

            records_file = os.path.join(
                RECORDS_DIR,
                f"{dataset_key}.json"
            )

            if not os.path.exists(records_file):
                print(
                    f"  Skipping {dataset_key}: "
                    f"records not found"
                )
                continue

            with open(
                records_file,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            records = data.get("all_records", [])

            lang = dataset_key.split("_")[0]

            print(
                f"  Processing transcripts for "
                f"{dataset_key} (lang={lang})"
            )

            preprocess_transcripts(
                lang,
                records,
                dataset_key
            )

            before = len(records)

            records = [
                r for r in records
                if r.get("text", "").strip()
            ]

            removed = before - len(records)

            if removed:
                print(
                    f"    Removed {removed} empty transcripts"
                )

            data["all_records"] = records

            if data.get("is_predetermined", False):

                if dataset_key == "en_cv_spon":
                    split_records = assign_by_membership(
                        records,
                        data["split_membership"]
                    )
                else:
                    split_records = split_data(records)

                split_source = "predetermined_full"

            else:
                split_records = split_data(records)
                split_source = "8_1_1"

            save_manifest(
                dataset_key,
                lang,
                split_source,
                "Transcript preprocessing applied",
                split_records,
                manifest_dir=original_dir
            )

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Balancing/Re-splitting
    # ─────────────────────────────────────────────────────────────
    if start_stage <= 4 <= end_stage:
        print("\n" + "=" * 70)
        print("STAGE 4: Balancing/Re-splitting")
        print("=" * 70)

        balanced_records = balance_lang_data(
            original_dir=original_dir,
            balanced_dir=balanced_dir,
            short_duration_threshold=3
        )

        for dataset_key, balanced_recs in balanced_records.items():

            manifest_path = os.path.join(
                original_dir,
                f"{dataset_key}_manifest.json"
            )

            if not os.path.exists(manifest_path):
                continue

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as f:
                manifest = json.load(f)

            lang = manifest["lang"]
            split_source = manifest["split_source"]

            original_records = (
                manifest.get("train", [])
                + manifest.get("dev", [])
                + manifest.get("test", [])
            )

            split_records = split_data(balanced_recs)

            orig_count = len(original_records)
            new_count = len(balanced_recs)

            orig_hours = calculate_hours(original_records)
            new_hours = calculate_hours(balanced_recs)

            train_h = round(
                calculate_hours(split_records["train"]),
                2
            )
            dev_h = round(
                calculate_hours(split_records["dev"]),
                2
            )
            test_h = round(
                calculate_hours(split_records["test"]),
                2
            )

            split_note = (
                f"Balanced: "
                f"{orig_count}→{new_count} utts, "
                f"{orig_hours:.1f}→{new_hours:.1f}h, "
                f"train={train_h}h "
                f"dev={dev_h}h "
                f"test={test_h}h"
            )

            save_manifest(
                dataset_key,
                lang,
                split_source,
                split_note,
                split_records,
                manifest_dir=balanced_dir
            )
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
