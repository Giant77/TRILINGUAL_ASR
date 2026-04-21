from preprocessing import *

def main():
    """
    Main pipeline:
      1. Process all datasets (in order by language)
      2. Calculate hours per language
      3. Balance data across languages
      4. Save final manifests
    """
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE START")
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
    
    # Step 1: Process all datasets and collect metadata
    print("\n" + "="*70)
    print("STEP 1: Processing all datasets")
    print("="*70)
    
    lang_datasets = {}  # {dataset_key: {'all_records': [...], 'is_predetermined': ...}}
    
    for lang, processor_func in processors:
        dataset_key = processor_func.__name__.replace('process_', '')
        result = processor_func()
        lang_datasets[dataset_key] = result
    
    # Step 2: Balance data across languages
    print("\n" + "="*70)
    print("STEP 2: Balancing data across languages")
    print("="*70)
    
    balanced_records = balance_lang_data(lang_datasets)
    
    # Step 3: Re-split balanced data and save manifests
    print("\n" + "="*70)
    print("STEP 3: Re-splitting balanced data and saving manifests")
    print("="*70)
    
    for dataset_key, original_result in lang_datasets.items():
        lang = dataset_key.split('_')[0]
        split_type = original_result.get('split_type', 'unknown')
        is_predetermined = original_result['is_predetermined']
        split_source = {True: 'predetermined_full', False: '8_1_1'}.get(is_predetermined, '8_1_1')
        
        # Get balanced records
        balanced_recs = balanced_records.get(dataset_key, original_result['all_records'])
        
        # Re-split balanced data
        if is_predetermined and split_type == 'full':
            # Full predetermined - use original splits from balanced data
            # (This is a simplification; ideally you'd preserve original split assignments)
            split_records = split_data(balanced_recs)
        elif is_predetermined and split_type == 'partial':
            # Partial predetermined - carve dev from balanced data
            split_records = split_data(balanced_recs)  # Simplified; use split function
        else:
            # 8:1:1 split
            split_records = split_data(balanced_recs)
        
        # Re-save manifest with balanced data
        split_note = original_result.get('split_records', {})
        save_manifest(dataset_key, lang, split_source,
                      'Balanced data split', split_records)
    
    print("\n" + "="*70)
    print(f"All manifests saved to: {MANIFEST_DIR}")
    print("Next: run local/manifests_to_kaldi.py")
    print("="*70)

if __name__ == "__main__":
    main()

# Run via windows/powershell:
# overwrite : python main.py 2>&1 | Tee-Object -FilePath log/preprocessing_log.txt
# append    : python main.py 2>&1 | Tee-Object -FilePath log/preprocessing_log.txt -Append

# Run via linux:
# overwrite : python main.py 2>&1 | tee log/preprocessing_log.txt
# append    : python main.py 2>&1 | tee -a log/preprocessing_log.txt
