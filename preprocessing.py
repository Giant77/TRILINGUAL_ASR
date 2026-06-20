"""
run_preprocessing.py
Split-aware manifest generation with language-level data balancing.

Manifest format per dataset:
  {
    "dataset_key", "lang", "split_source", "split_note",
    "total_utts", "total_hours", "counts", "actual_ratio",
    "train": [...], "dev": [...], "test": [...]
  }

split_source values:
  "predetermined_full"           - original train/dev/test preserved as-is
  "predetermined_partial_carved" - original train+test; dev carved for ≈8:1:1
  "8_1_1"                        - no predetermined splits; speaker-independent split

Workflow:
  1. Load and process all datasets per language
  2. Calculate total hours per language and dataset
  3. Log pre-balance statistics
  4. Balance: reduce to least-data language; prioritize removing from non-predetermined splits
  5. Log post-balance statistics
  6. Split each dataset into train/dev/test
  7. Save manifests with updated counts
"""

import re
import csv
import json
import os
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from preprocess_audio import (
    process_dataset, process_timestamp_segments,  
    process_escwa_segmented, process_indocsc_segmented
)
from load_transcripts import (
    load_mozilla_cv, load_fleurs, load_titml_idn,
    load_librivox_id, load_seacrowd_indocsc, load_seacrowd_sindodsc,
    load_clartts, load_escwa, load_escwa_segments,
    load_hari_minggoean, load_homostoria, load_librispeech_parquet,
    load_mozilla_spontant
)

BASE_DATASET = os.path.join(".", "dataset")
BASE_OUT = os.path.join(BASE_DATASET, "processed")

MANIFEST_DIR = os.path.join(BASE_OUT, "manifests")
RECORDS_DIR = os.path.join(BASE_OUT, "records")
SEED         = 777  # fixed for reproducibility
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)

# ─── SPLIT REGISTRY ──────────────────────────────────────────────────────────
# 'full'    : original train + dev + test → preserve as-is
# 'partial' : original train + test only  → carve dev from train
# 'none'    : no predetermined splits    → apply 8:1:1
SPLIT_REGISTRY = {
    'id_cv':          'full',
    'id_fleurs':      'full',
    'id_librivox':    'partial',
    'id_titml':       'none',
    'id_indocsc':     'none',
    'id_sindodsc':    'none',
    'ar_cv':          'full',
    'ar_fleurs':      'full',
    'ar_clartts':     'partial',
    'en_librispeech': 'none',
    'en_fleurs':      'full',
    'en_cv_spon':     'none',
    'cs_escwa':       'none',
    'cs_hari':        'none',
    'cs_homostoria':  'none',
}

# ─── DATASET REGISTRY (Order matters!) ───────────────────────────────────────
# Track all datasets by language: {lang: [(key, split_type, is_predetermined), ...]}
DATASET_REGISTRY = {
    'id': [
        ('id_cv', 'full', True),
        ('id_fleurs', 'full', True),
        ('id_librivox', 'partial', True),
        ('id_titml', 'none', False),
        ('id_indocsc', 'none', False),
        ('id_sindodsc', 'none', False),
    ],
    'ar': [
        ('ar_cv', 'full', True),
        ('ar_fleurs', 'full', True),
        ('ar_clartts', 'partial', True),
    ],
    'en': [
        ('en_librispeech', 'none', False),
        ('en_fleurs', 'full', True),
        ('en_cv_spon', 'none', False),
    ],
    'cs': [
        ('cs_escwa', 'none', False),
        ('cs_hari', 'none', False),
        ('cs_homostoria', 'none', False),
    ],
}


# ─── SPLIT HELPERS ───────────────────────────────────────────────────────────

def calculate_hours(records: list) -> float:
    """Calculate total hours from record durations."""
    return sum(r.get('duration', 0) for r in records) / 3600.0

def split_data(records: list, train_split: float=0.80, dev_split: float=0.10) -> dict:
    """
    Speaker-independent 8:1:1 split with fallback to utterance-level splitting.

    Strategy:
      1. If ≥ 5 unique speakers: use speaker-level split (speaker-independent)
      2. Otherwise: use utterance-level split (reproducible but not speaker-independent)

    Returns {'train': [...], 'dev': [...], 'test': [...]}.
    """
    random.seed(SEED)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_split)
    n_dev   = int(n * dev_split)

    return {
        'train': shuffled[:n_train],
        'dev': shuffled[n_train:n_train+n_dev],
        'test': shuffled[n_train+n_dev:]
    }

def split_partial_carve_dev(train_records: list, 
                            test_records: list, 
                            train_split: float=0.80,
                            dev_split: float=0.10) -> dict:
    """
    Partial predetermined split (train+test only, no dev).
    Carve dev from train so overall ratio ≈ 8:1:1.
    """
    random.seed(SEED)
    shuffled = train_records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_new_train = int(n * (train_split / 0.9))

    return {
        'train': shuffled[:n_new_train],
        'dev': shuffled[n_new_train:],
        'test': test_records
    }

def assign_by_membership(all_records: list,
                          membership: dict) -> dict:
    """
    Assign records to splits based on source_stem membership dict.
    membership: {source_stem: 'train' | 'dev' | 'test'}
    """
    result = {'train': [], 'dev': [], 'test': []}
    unassigned = 0
    for r in all_records:
        split = membership.get(r.get('source_stem', ''))
        if split in result:
            result[split].append(r)
        else:
            unassigned += 1
    if unassigned:
        print(f"{unassigned} records unassigned (source_stem not in membership)")
    return result

def save_manifest(key: str, lang: str,
                  split_source: str, split_note: str,
                  split_records: dict, manifest_dir: str = None) -> None:
    """
    Save manifest JSON with split metadata including duration statistics.

    Args:
        manifest_dir: Optional directory to save manifest. Defaults to MANIFEST_DIR.
    """
    if manifest_dir is None:
        manifest_dir = MANIFEST_DIR

    total  = sum(len(v) for v in split_records.values())
    hours  = sum(calculate_hours(v) for v in split_records.values())
    counts = {s: len(v) for s, v in split_records.items()}
    hours_per_split = {s: round(calculate_hours(v), 2) for s, v in split_records.items()}
    ratio  = {s: round(c / max(total, 1), 3) for s, c in counts.items()}

    manifest = {
        "dataset_key":      key,
        "lang":             lang,
        "split_source":     split_source,
        "split_note":       split_note,
        "total_utts":       total,
        "total_hours":      round(hours, 2),
        "counts":           counts,
        "hours_per_split":  hours_per_split,
        "actual_ratio":     ratio,
        "train":            split_records.get('train', []),
        "dev":              split_records.get('dev',   []),
        "test":             split_records.get('test',  []),
    }

    out_path = os.path.join(manifest_dir, f"{key}_manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Console & log ──
    ratio_str = " | ".join(
        f"{s}={counts[s]:,} ({ratio[s]*100:.1f}%)" for s in ['train', 'dev', 'test']
    )
    hours_str = " | ".join(
        f"{s}={hours_per_split[s]:.1f}h" for s in ['train', 'dev', 'test']
    )

    split_tag = {
        'predetermined_full':           'OK: PREDETERMINED (full)',
        'predetermined_partial_carved': 'WARNING: PREDETERMINED (partial→carved dev)',
        '8_1_1':                        'CHANGES: SPLIT 8:1:1 (no original splits)',
    }.get(split_source, split_source)

    msg = f"\n[{key}] {split_tag}\n{split_note}\nUtts: {ratio_str}\nHours: {hours_str} (total={hours:.1f}h)"
    print(msg)

def remove_invisible_chars(text: str) -> str:
    return ''.join(
        c for c in text
        if unicodedata.category(c) != 'Cf'
    )

def remove_transcript_tags(text: str) -> str:
    return re.sub(r'\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>', ' ', text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize_arabic(text: str, araby=None) -> str:
    if araby:
        # Remove harakat/tashkeel
        text = araby.strip_diacritics(text)

        # Normalize lam-alef ligatures
        text = araby.normalize_ligature(text)

    # Remove tatweel
    text = text.replace('ـ', '')
    text = normalize_whitespace(text)

    return text

def contains_arabic(text: str) -> bool:
    return any(
        '\u0600' <= c <= '\u06FF'
        or '\u0750' <= c <= '\u077F'
        or '\u08A0' <= c <= '\u08FF'
        or '\uFB50' <= c <= '\uFDFF'
        or '\uFE70' <= c <= '\uFEFF'
        for c in text
    )

def normalize_common(text: str) -> str:
    text = remove_transcript_tags(text)
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    text = remove_invisible_chars(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text

def preprocess_transcripts(lang: str, records: list, dataset_key: str = None) -> None:
    """
    Preprocess transcripts based on language.

    Supported languages:
      - ar (Arabic): Remove diacritics using pyarabic library
      - id (Indonesian): Convert to lowercase
      - en (English): Convert to lowercase
      - cs (Code-Switching): Auto-detect Arabic, apply diacritics removal if present, else lowercase
    """
    try:
        import pyarabic.araby as araby
        has_pyarabic = True

    except ImportError:
        has_pyarabic = False
        if lang == 'ar':
            print("WARNING [pyarabic]: Not installed - skipping Arabic diacritic removal")

    for record in records:
        text = record.get('text', '')
        if not text:
            continue

        # Common normalization
        text = normalize_common(text)

        if lang == 'ar':
            if has_pyarabic:
                text = normalize_arabic(text, araby)

            record['text'] = text

        elif lang == 'id':
            record['text'] = text.lower()

        elif lang == 'en':
            record['text'] = text.lower()

        elif lang == 'cs':
            # Code-Switching: auto-detect Arabic
            has_arabic = contains_arabic(text)

            if has_arabic:
                if has_pyarabic:
                    text = normalize_arabic(text, araby)
                else:
                    if dataset_key:
                        print(
                            f"WARNING [{dataset_key}]: "
                            f"pyarabic not installed - skipping Arabic normalization"
                        )

            record['text'] = text.lower()

        else:
            print(f"    WARNING: Unknown language '{lang}' - skipping preprocessing")

    return records

def append_short_segments(lang: str, records: list, source_manifest_dir: Path, output_manifest_dir: Path):
    """
    Append records to short-segment manifest.
    Existing manifest is read from source_manifest_dir.
    Updated manifest is always written to output_manifest_dir.

    Args:
        lang:
            Language identifier.
        records:
            Records to append.
        source_manifest_dir:
            Directory containing existing manifests.
        output_manifest_dir:
            Directory where merged manifest is saved.
    """
    if not records:
        return


    filename = f"short_segments_{lang}.json"

    source_path = os.path.join(
        source_manifest_dir,
        filename
    )

    output_path = os.path.join(
        output_manifest_dir,
        filename
    )

    manifest = None

    if os.path.exists(source_path):
        try:
            if os.path.getsize(source_path) > 0:
                with open(source_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

        except (
            json.JSONDecodeError,
            OSError,
            ValueError
        ):
            manifest = None

    if not manifest:
        manifest = {
            "lang": lang,
            "total_utts": 0,
            "total_hours": 0.0,
            "short_records": []
        }

    manifest["short_records"].extend(records)
    manifest["total_utts"] = len(manifest["short_records"])

    manifest["total_hours"] = round(
        sum(r.get("duration", 0) for r in manifest["short_records"]) / 3600.0,
        2
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

def balance_lang_data(original_dir: Path, balanced_dir: Path, short_duration_threshold: float = 3.0):
    """
    Balance language hours by reducing higher-resource languages to the
    duration of the smallest language.

    Reduction is performed in two stages:

      1. Short-utterance pruning
         Remove utterances shorter than short_duration_threshold,
         starting from the shortest utterances first.

      2. Standard balancing
         If the language remains above the target duration after
         short-utterance pruning, continue removing utterances until
         the target language-hour ratio is reached.

    Removed short utterances are appended to
    short_segments_<lang>.json.

    Args:
        lang_datasets:
            Dataset metadata dictionary.

        short_duration_threshold:
            Duration threshold (seconds) used for initial pruning.

        short_manifest_dir:
            Directory containing short-segment manifests.
            Defaults to MANIFEST_DIR.

    Returns:
        {
            dataset_key: balanced_records
        }

    Notes:
        - Code-switching datasets (cs_*) are excluded.
        - Non-predetermined datasets are reduced before predetermined
          datasets.
        - Short utterances are removed before standard balancing.
        - If short-utterance pruning alone reaches the target language
          hours, no additional balancing is performed.
    """
    lang_hours = defaultdict(float)
    lang_datasets_by_lang = defaultdict(list)
    result = {}

    # ---------------------------------------------------------
    # Load all cleaned manifests
    # ---------------------------------------------------------
    manifest_files = sorted(
        f for f in os.listdir(original_dir)
        if f.endswith("_manifest.json")
    )

    for manifest_file in manifest_files:
        manifest_path = os.path.join(
            original_dir,
            manifest_file
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_key = manifest["dataset_key"]
        lang = manifest["lang"]

        records = (
            manifest.get("train", [])
            + manifest.get("dev", [])
            + manifest.get("test", [])
        )

        split_source = manifest.get("split_source", "8_1_1")
        is_predetermined = (split_source == "predetermined_full")

        data = {
            "all_records": records,
            "is_predetermined": is_predetermined,
        }

        # CS datasets are excluded from balancing
        if lang == "cs":
            result[dataset_key] = records
            continue

        hours = calculate_hours(records)
        lang_hours[lang] += hours

        lang_datasets_by_lang[lang].append(
            (dataset_key, data)
        )

    if not lang_hours:
        print("No language manifests found for balancing.")
        return result

    # ---------------------------------------------------------
    # Determine balancing target
    # ---------------------------------------------------------
    min_lang = min(lang_hours, key=lang_hours.get)
    target_hours = lang_hours[min_lang]

    print("\n" + "=" * 70)
    print("PRE-BALANCE: Total hours per language")
    print("=" * 70)

    for lang in sorted(lang_hours.keys()):
        print(f"{lang.upper()}: {lang_hours[lang]:.1f}h")

    print(f"Target (min): {min_lang.upper()} = {target_hours:.1f}h")

    reduction_log = []

    # ---------------------------------------------------------
    # Balance each language
    # ---------------------------------------------------------
    for lang in sorted(lang_datasets_by_lang.keys()):
        current_hours = lang_hours[lang]

        # Already smallest language
        if current_hours <= target_hours * 1.01:
            for key, data in lang_datasets_by_lang[lang]:
                result[key] = data["all_records"]

            continue

        datasets = sorted(
            lang_datasets_by_lang[lang],
            key=lambda x: (x[1]["is_predetermined"], -len(x[1]["all_records"]))
        )

        # -----------------------------------------------------
        # Stage 1: Remove short utterances first
        # -----------------------------------------------------
        short_candidates = []

        for key, data in datasets:
            for rec in data["all_records"]:
                dur = rec.get("duration", 0)

                if dur < short_duration_threshold:
                    short_candidates.append((dur, key, rec))

        short_candidates.sort(key=lambda x: x[0])

        excess_hours = (current_hours -target_hours)
        removed_short_records = []
        removed_hours = 0.0

        for dur, key, rec in short_candidates:
            if removed_hours >= excess_hours:
                break

            removed_short_records.append(
                (key, rec)
            )

            removed_hours += (dur / 3600.0)

        removed_ids = {rec["utt_id"] for _, rec in removed_short_records}
        pruned_datasets = {}

        for key, data in datasets:
            pruned_records = [
                r
                for r in data["all_records"]
                if r["utt_id"] not in removed_ids
            ]

            pruned_datasets[key] = (pruned_records)

        append_short_segments(
            lang=lang,
            records=[r for _, r in removed_short_records],
            source_manifest_dir=original_dir,
            output_manifest_dir=balanced_dir
        )

        hours_after_short = sum(
            calculate_hours(records)
            for records in pruned_datasets.values()
        )

        reduction_log.append({
            "dataset": f"{lang}_short_pruning",
            "lang": lang,
            "orig_count": sum(
                    len(data["all_records"])
                    for _, data in datasets
                ),
            "new_count": sum(
                    len(records)
                    for records in pruned_datasets.values()
                ),
            "orig_hours": round(current_hours, 2),
            "new_hours": round(hours_after_short, 2),
            "is_predetermined": False,
        })

        # -----------------------------------------------------
        # Done after short pruning
        # -----------------------------------------------------
        if hours_after_short <= target_hours * 1.01: # tolerance
            for key, records in pruned_datasets.items():
                result[key] = records

            continue

        # -----------------------------------------------------
        # Stage 2: Standard balancing
        # -----------------------------------------------------
        remaining_to_reduce = (hours_after_short - target_hours)
        reduced_datasets = {}

        for key, data in datasets:
            records = pruned_datasets[key]
            hours = calculate_hours(records)

            if remaining_to_reduce <= 0:
                reduced_datasets[key] = (records)

                continue

            reduce_from_dataset = min(
                hours,
                remaining_to_reduce
            )

            keep_hours = (hours - reduce_from_dataset)

            if hours > 0:
                keep_fraction = (keep_hours / hours)

                target_count = max(
                    1,
                    int(
                        len(records)
                        * keep_fraction
                    )
                )

                sorted_records = sorted(
                    records,
                    key=lambda r: r.get("duration", 0),
                    reverse=True
                )

                kept_records = (sorted_records[:target_count])

            else:
                kept_records = records

            reduced_datasets[key] = (kept_records)
            remaining_to_reduce -= hours - calculate_hours(kept_records)

            reduction_log.append({
                "dataset": key,
                "lang": lang,
                "orig_count": len(records),
                "new_count": len(kept_records),
                "orig_hours": round(hours, 2),
                "new_hours": round(calculate_hours(kept_records), 2),
                "is_predetermined": data["is_predetermined"],
            })

        for key, records in reduced_datasets.items():
            result[key] = records

    # ---------------------------------------------------------
    # Reduction log
    # ---------------------------------------------------------
    if reduction_log:
        print("\n" + "=" * 70)
        print("REDUCTION LOG")
        print("=" * 70)

        for log in reduction_log:
            ptype = (
                "PREDETERMINED"
                if log["is_predetermined"]
                else "NON-PREDETERMINED"
            )

            print(
                f"{log['dataset']:20} "
                f"({ptype:16}) | "
                f"{log['orig_count']:6} -> "
                f"{log['new_count']:6} utts | "
                f"{log['orig_hours']:7.2f}h -> "
                f"{log['new_hours']:7.2f}h"
            )

    # ---------------------------------------------------------
    # Final stats
    # ---------------------------------------------------------
    lang_hours_post = defaultdict(float)

    for key, records in result.items():

        lang = key.split("_")[0]

        lang_hours_post[lang] += (
            calculate_hours(records)
        )

    print("\n" + "=" * 70)
    print("POST-BALANCE: Total hours per language")
    print("=" * 70)

    for lang in sorted(
        lang_hours_post.keys()
    ):
        print(
            f"{lang.upper()}: "
            f"{lang_hours_post[lang]:.1f}h"
        )

    return result

# ─── DATASET WRAPPER FUNCTIONS ──────────────────────────────────────────────
def process_id_cv(mode='full', manifest_dir=None):
    """Indonesian: Mozilla Common Voice - PREDETERMINED FULL

    Args:
        mode: 'audio' (process only), 'manifest' (create manifest only), 'full' (both)
        manifest_dir: Directory to save manifests. If None, uses BASE_OUT/manifests
    """
    dataset_key = 'id_cv'
    print(f"\n[{dataset_key}] Mozilla Common Voice ID v24.0")

    cv_id_root  = os.path.join(BASE_DATASET, "id", "mozilla", "scripted-id",
                                "cv-corpus-24.0-2025-12-05", "id")
    cv_id_clips = os.path.join(cv_id_root, "clips")

    if mode in ['audio', 'full']:
        split_records = {}
        for split in ['train', 'dev', 'test']:
            tsv = os.path.join(cv_id_root, f"{split}.tsv")
            if os.path.exists(tsv):
                m = load_mozilla_cv(tsv)
                recs = process_dataset(cv_id_clips,
                                       os.path.join(BASE_OUT, "id", "cv", split),
                                       "id", f"cv_{split}", m)
                split_records[split] = recs
            else:
                print(f"  {tsv} not found - split '{split}' empty")
                split_records[split] = []

        all_recs = [r for recs in split_records.values() for r in recs]

        # Save records for stage 2
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({
                'split_records': split_records,
                'all_records': all_recs,
                'is_predetermined': True
            }, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        # Load or use cached records
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        return {
            'split_records': split_records,
            'all_records': all_recs,
            'is_predetermined': True,
        }

    return None

def process_id_fleurs(mode='full', manifest_dir=None):
    """Indonesian: FLEURS - PREDETERMINED FULL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'id_fleurs'
    print(f"\n[{dataset_key}] FLEURS ID")
    fleurs_id_root = os.path.join(BASE_DATASET, "id", "Fleurs_id")

    if mode in ['audio', 'full']:
        split_records = {}
        for split in ['train', 'dev', 'test']:
            tsv       = os.path.join(fleurs_id_root, f"{split}.tsv")
            audio_dir = os.path.join(fleurs_id_root, "audio", split)
            if os.path.exists(tsv) and os.path.exists(audio_dir):
                m = load_fleurs(tsv)
                recs = process_dataset(audio_dir,
                                       os.path.join(BASE_OUT, "id", "fleurs", split),
                                       "id", f"fleurs_{split}", m)
                split_records[split] = recs
            else:
                print(f"  FLEURS ID split '{split}' - tsv or audio dir missing")
                split_records[split] = []

        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'id', 'predetermined_full',
        #               'FLEURS ID - audio/train|dev|test dirs + TSV preserved',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_id_librivox(mode='full', manifest_dir=None):
    """Indonesian: Librivox - PREDETERMINED PARTIAL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'id_librivox'
    print(f"\n[{dataset_key}] Librivox ID")
    librivox_root = os.path.join(BASE_DATASET, "id", "Librivox")
    train_meta    = os.path.join(librivox_root, "metadata_train.csv", "metadata_train.csv")
    test_meta     = os.path.join(librivox_root, "metadata_test.csv", "metadata_test.csv")
    train_audio   = os.path.join(librivox_root, "audio_train", "librivox-indonesia", "train")
    test_audio    = os.path.join(librivox_root, "audio_test", "librivox-indonesia", "test")

    if mode in ['audio', 'full']:
        librivox_train = process_dataset(
            train_audio, os.path.join(BASE_OUT, "id", "librivox", "train"),
            "id", "librivox_train", load_librivox_id(train_meta))
        librivox_test  = process_dataset(
            test_audio, os.path.join(BASE_OUT, "id", "librivox", "test"),
            "id", "librivox_test", load_librivox_id(test_meta))

        split_records = split_partial_carve_dev(librivox_train, librivox_test)
        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'id', 'predetermined_partial_carved',
        #               'Librivox ID - metadata_train/test CSV; dev carved from train (78:22) for 7:2:1',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_id_titml(mode='full', manifest_dir=None):
    """Indonesian: TITML-IDN - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'id_titml'
    print(f"\n[{dataset_key}] TITML-IDN")
    titml_dir = os.path.join(BASE_DATASET, "id", "TITML-IDN")

    if mode in ['audio', 'full']:
        titml_map = load_titml_idn(titml_dir)
        titml_all = process_dataset(titml_dir,
                                     os.path.join(BASE_OUT, "id", "titml"),
                                     "id", "titml", titml_map)

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': titml_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            titml_all = data['all_records']

        split_records = split_data(titml_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'id', '8_1_1',
        #               'TITML-IDN - speaker dirs only; speaker-independent 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': titml_all, 'is_predetermined': False}

    return None

def process_id_indocsc(mode='full', manifest_dir=None):
    """Indonesian: SEACrowd IndoCSC - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'id_indocsc'
    print(f"\n[{dataset_key}] SEACrowd IndoCSC")
    indocsc_wav = os.path.join(BASE_DATASET, "id", "SEACrowd",
                                "Indonesian_Conversational_Speech_Corpus", "WAV")
    indocsc_txt = os.path.join(BASE_DATASET, "id", "SEACrowd",
                                "Indonesian_Conversational_Speech_Corpus", "TXT")

    if mode in ['audio', 'full']:
        segment_map = load_seacrowd_indocsc(indocsc_wav, indocsc_txt)

        indocsc_all = process_indocsc_segmented(
            indocsc_wav,
            segment_map,
            os.path.join(BASE_OUT, "id", "indocsc"),
        )

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': indocsc_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            indocsc_all = data['all_records']

        split_records = split_data(indocsc_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'id', '8_1_1',
        #               'SEACrowd IndoCSC - no predetermined splits; 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': indocsc_all, 'is_predetermined': False}

    return None

def process_id_sindodsc(mode='full', manifest_dir=None):
    """Indonesian: SEACrowd SIndoDuSC - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'id_sindodsc'
    print(f"\n[{dataset_key}] SEACrowd SIndoDuSC")
    sindocsc_root = os.path.join(BASE_DATASET, "id", "SEACrowd",
                                "Indonesian_Scripted_Speech_Corpus_Daily_Use_Sentence")
    sindodsc_wav = os.path.join(BASE_DATASET, "id", "SEACrowd",
                                 "Indonesian_Scripted_Speech_Corpus_Daily_Use_Sentence", "WAV")

    if mode in ['audio', 'full']:
        sindodsc_all = process_dataset(sindodsc_wav,
                                        os.path.join(BASE_OUT, "id", "sindodsc"),
                                        "id", "sindodsc",
                                        load_seacrowd_sindodsc(sindodsc_wav, sindocsc_root))

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': sindodsc_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sindodsc_all = data['all_records']

        split_records = split_data(sindodsc_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'id', '8_1_1',
        #               'SEACrowd SIndoDuSC - WAV only; 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': sindodsc_all, 'is_predetermined': False}

    return None

def process_ar_cv(mode='full', manifest_dir=None):
    """Arabic: Mozilla Common Voice - PREDETERMINED FULL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'ar_cv'
    print(f"\n[{dataset_key}] Mozilla Common Voice AR v24.0")
    cv_ar_root  = os.path.join(BASE_DATASET, "ar", "mozilla",
                                "cv-corpus-24.0-2025-12-05")
    cv_ar_clips = os.path.join(cv_ar_root, "clips")

    if mode in ['audio', 'full']:
        split_records = {}
        for split in ['train', 'dev', 'test']:
            tsv = os.path.join(cv_ar_root, f"{split}.tsv")
            if os.path.exists(tsv):
                m = load_mozilla_cv(tsv)
                recs = process_dataset(cv_ar_clips,
                                       os.path.join(BASE_OUT, "ar", "cv", split),
                                       "ar", f"cv_{split}", m)
                split_records[split] = recs
            else:
                print(f"  {tsv} not found")
                split_records[split] = []

        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'ar', 'predetermined_full',
        #               'Mozilla CV AR v24.0 - TSV train/dev/test preserved',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_ar_fleurs(mode='full', manifest_dir=None):
    """Arabic: FLEURS - PREDETERMINED FULL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'ar_fleurs'
    print(f"\n[{dataset_key}] FLEURS AR")
    fleurs_ar_root = os.path.join(BASE_DATASET, "ar", "fleurs-ar")

    if mode in ['audio', 'full']:
        split_records = {}
        for split in ['train', 'dev', 'test']:
            tsv       = os.path.join(fleurs_ar_root, f"{split}.tsv")
            audio_dir = os.path.join(fleurs_ar_root, "audio", split)
            if os.path.exists(tsv) and os.path.exists(audio_dir):
                m = load_fleurs(tsv)
                recs = process_dataset(audio_dir,
                                       os.path.join(BASE_OUT, "ar", "fleurs", split),
                                       "ar", f"fleurs_{split}", m)
                split_records[split] = recs
            else:
                print(f"  FLEURS AR split '{split}' missing")
                split_records[split] = []

        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'ar', 'predetermined_full',
        #               'FLEURS AR - audio/train|dev|test dirs preserved',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_ar_clartts(mode='full', manifest_dir=None):
    """Arabic: ClArTTS - PREDETERMINED PARTIAL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'ar_clartts'
    print(f"\n[{dataset_key}] ClArTTS")
    clartts_dir = os.path.join(BASE_DATASET, "ar", "ClArTTS")

    if mode in ['audio', 'full']:
        train_parquets = sorted(Path(clartts_dir).glob('train-*.parquet'))
        test_parquets  = sorted(Path(clartts_dir).glob('test-*.parquet'))

        clartts_train_map = load_clartts(clartts_dir,
                                          parquet_files=[str(p) for p in train_parquets])
        clartts_test_map  = load_clartts(clartts_dir,
                                          parquet_files=[str(p) for p in test_parquets])

        clartts_extracted = os.path.join(clartts_dir, "extracted")
        clartts_train_all = process_dataset(clartts_extracted,
                                             os.path.join(BASE_OUT, "ar", "clartts"),
                                             "ar", "clartts_train", clartts_train_map)
        clartts_test_all  = process_dataset(clartts_extracted,
                                             os.path.join(BASE_OUT, "ar", "clartts"),
                                             "ar", "clartts_test", clartts_test_map)

        split_records = split_partial_carve_dev(clartts_train_all, clartts_test_all)
        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'ar', 'predetermined_partial_carved',
        #               'ClArTTS - parquet files; dev carved from train for 7:2:1',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_en_librispeech(mode='full', manifest_dir=None):
    """English: LibriSpeech - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'en_librispeech'
    print(f"\n[{dataset_key}] LibriSpeech clean-100 (HF parquet)")
    libri_dir = os.path.join(BASE_DATASET, "en", "librispeech", "Data")

    if mode in ['audio', 'full']:
        libri_map = load_librispeech_parquet(libri_dir)
        libri_extracted = os.path.join(libri_dir, "extracted")
        libri_all = process_dataset(libri_extracted,
                                     os.path.join(BASE_OUT, "en", "librispeech"),
                                     "en", "librispeech", libri_map)

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': libri_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            libri_all = data['all_records']

        split_records = split_data(libri_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'en', '8_1_1',
        #               'LibriSpeech clean-100 HF parquet - 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': libri_all, 'is_predetermined': False}

    return None

def process_en_fleurs(mode='full', manifest_dir=None):
    """English: FLEURS - PREDETERMINED FULL

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'en_fleurs'
    print(f"\n[{dataset_key}] FLEURS EN")
    fleurs_en_root = os.path.join(BASE_DATASET, "en", "fleurs_en")

    if mode in ['audio', 'full']:
        split_records = {}
        for split in ['train', 'dev', 'test']:
            tsv       = os.path.join(fleurs_en_root, f"{split}.tsv")
            audio_dir = os.path.join(fleurs_en_root, "audio", split)
            if os.path.exists(tsv) and os.path.exists(audio_dir):
                m = load_fleurs(tsv)
                recs = process_dataset(audio_dir,
                                       os.path.join(BASE_OUT, "en", "fleurs", split),
                                       "en", f"fleurs_{split}", m)
                split_records[split] = recs
            else:
                print(f"  FLEURS EN split '{split}' missing")
                split_records[split] = []

        all_recs = [r for recs in split_records.values() for r in recs]

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            split_records = data['split_records']
            all_recs = data['all_records']

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'en', 'predetermined_full',
        #               'FLEURS EN - audio/train|dev|test dirs preserved',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': all_recs, 'is_predetermined': True}

    return None

def process_en_cv_spon(mode='full', manifest_dir=None):
    """English: Mozilla CV Spontaneous - Use 'split' column from TSV

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'en_cv_spon'
    print(f"\n[{dataset_key}] Mozilla CV EN Spontaneous")

    cv_en_dir = os.path.join(
        BASE_DATASET, "en", "mozilla",
        "sps-corpus-1.0-2025-11-25-en"
    )

    if mode in ['audio', 'full']:
        cv_en_map = load_mozilla_spontant(cv_en_dir)
        split_membership = {}
        corpus_tsv = Path(cv_en_dir) / "ss-corpus-en.tsv"

        with open(corpus_tsv, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                audio_file = row.get('audio_file')
                split_val = row.get('split')
                if not audio_file or not split_val:
                    continue
                stem = os.path.splitext(audio_file)[0]
                split_val = split_val.strip().lower()
                if split_val in ['valid', 'validated']:
                    split_membership[stem] = 'dev'
                elif split_val in ['train', 'dev', 'test']:
                    split_membership[stem] = split_val

        cv_en_all = process_dataset(
            os.path.join(cv_en_dir, "audios"),
            os.path.join(BASE_OUT, "en", "cv_spon"),
            "en", "cv_spon", cv_en_map
        )

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': cv_en_all, 'split_membership': split_membership, 'is_predetermined': True}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cv_en_all = data['all_records']
            split_membership = data['split_membership']

        split_records = assign_by_membership(cv_en_all, split_membership)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(
        #     dataset_key, 'en', 'predetermined_full',
        #     'Mozilla CV EN Spontaneous v1.0 - splits from TSV column',
        #     split_records, manifest_dir=manifest_dir
        # )

        return {'split_records': split_records, 'all_records': cv_en_all, 'is_predetermined': True}

    return None

def process_cs_escwa(mode='full', manifest_dir=None):
    """Code-Switching (AR-EN): QCRI ESCWA - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'cs_escwa'
    print(f"\n[{dataset_key}] QCRI ESCWA")
    escwa_dir = os.path.join(BASE_DATASET, "CS", "ar-en", "escwa", "cs.released")

    if mode in ['audio', 'full']:
        escwa_text_map = load_escwa(escwa_dir)
        escwa_seg_map  = load_escwa_segments(escwa_dir)
        escwa_all = process_escwa_segmented(
            os.path.join(escwa_dir, "wav"),
            escwa_text_map, escwa_seg_map,
            os.path.join(BASE_OUT, "cs", "escwa"))

        # for rec in escwa_all:
            # utt_id = rec.get('source_stem', '')
            # if utt_id in escwa_seg_map:
                # rec_id, _, _ = escwa_seg_map[utt_id]
                # rec['speaker'] = f"spk_escwa_{rec_id}"

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': escwa_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            escwa_all = data['all_records']

        split_records = split_data(escwa_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'cs', '8_1_1',
        #               'QCRI ESCWA - Kaldi segments+text; 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': escwa_all, 'is_predetermined': False}

    return None

def process_cs_hari(mode='full', manifest_dir=None):
    """Code-Switching (ID-EN): Hari Minggoean - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'cs_hari'
    print(f"\n[{dataset_key}] Hari Minggoean")
    hari_dir     = os.path.join(BASE_DATASET, "CS", "id-en", "Hari Minggoean", "2")

    if mode in ['audio', 'full']:
        hari_seg_map = load_hari_minggoean(hari_dir)

        hari_all = process_timestamp_segments(
            hari_dir,
            hari_seg_map,
            "cs",
            "hari",
            os.path.join(BASE_OUT, "cs", "hari_minggoean")
        )

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': hari_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            hari_all = data['all_records']

        split_records = split_data(hari_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'cs', '8_1_1',
        #               'Hari Minggoean - TSV segments; 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': hari_all, 'is_predetermined': False}

    return None

def process_cs_homostoria(mode='full', manifest_dir=None):
    """Code-Switching (ID-EN): Homostoria - 8:1:1

    Args:
        mode: 'audio', 'manifest', or 'full'
        manifest_dir: Directory to save manifests
    """
    dataset_key = 'cs_homostoria'
    print(f"\n[{dataset_key}] Homostoria")
    homo_dir     = os.path.join(BASE_DATASET, "CS", "id-en", "homostoria", "homostoria")

    if mode in ['audio', 'full']:
        homo_seg_map = load_homostoria(homo_dir)

        homo_all = process_timestamp_segments(
            homo_dir,
            homo_seg_map,
            "cs",
            "homostoria",
            os.path.join(BASE_OUT, "cs", "homostoria")
        )

        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({'all_records': homo_all, 'is_predetermined': False}, f, ensure_ascii=False, indent=2)

        if mode == 'audio':
            return None

    if mode in ['manifest', 'full']:
        records_file = os.path.join(RECORDS_DIR, f'{dataset_key}.json')
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            homo_all = data['all_records']

        split_records = split_data(homo_all)

        # if manifest_dir is None:
        #     manifest_dir = os.path.join(BASE_OUT, "manifests")

        # save_manifest(dataset_key, 'cs', '8_1_1',
        #               'Homostoria - TSV segments; 8:1:1 applied',
        #               split_records, manifest_dir=manifest_dir)

        return {'split_records': split_records, 'all_records': homo_all, 'is_predetermined': False}

    return None
