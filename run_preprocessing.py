"""
run_preprocessing.py
Split-aware manifest generation.

Manifest format per dataset:
  {
    "dataset_key", "lang", "split_source", "split_note",
    "total_utts", "counts", "actual_ratio",
    "train": [...], "dev": [...], "test": [...]
  }

split_source values:
  "predetermined_full"           — original train/dev/test preserved as-is
  "predetermined_partial_carved" — original train+test; dev carved for ≈8:1:1
  "8_1_1"                        — no predetermined splits; speaker-independent split
"""

import io
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from preprocess_audio import process_dataset, CONFIG, get_duration
from load_transcripts import (
    load_mozilla_cv, load_fleurs, load_titml_idn,
    load_librivox_id, load_seacrowd_indocsc, load_seacrowd_sindodsc,
    load_clartts, load_escwa, load_escwa_segments,
    load_hari_minggoean, load_homostoria, load_librispeech_parquet,
)


# ─── PATHS — CHANGE THESE ────────────────────────────────────────────────────
BASE_DATASET = r"D:\FYP\Trilingual_ASR\Dataset"
BASE_OUT     = r"D:\FYP\Trilingual_ASR\Dataset\processed"

MANIFEST_DIR = os.path.join(BASE_OUT, "manifests")
SEED         = 42  # fixed for reproducibility
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ─── SPLIT REGISTRY ──────────────────────────────────────────────────────────
# 'full'    : original train + dev + test → preserve as-is
# 'partial' : original train + test only  → carve dev from train
# 'none'    : no predetermined splits    → apply 8:1:1 TODO: changes split
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


# ─── SPLIT HELPERS ───────────────────────────────────────────────────────────

def _speaker_shuffle(records: list) -> list:
    """Group by speaker, shuffle speakers (fixed seed), return records in shuffled order."""
    import random
    random.seed(SEED)
    spk_to_recs = defaultdict(list)
    for r in records:
        spk_to_recs[r.get('speaker', 'spk_unk')].append(r)
    speakers = sorted(spk_to_recs.keys())
    random.shuffle(speakers)
    return speakers, spk_to_recs


def split_data(records: list, train_split: float=0.80, dev_split: float=0.10) -> dict:
    """
    train_split   : train data ratio, default=0.80
    dev_split     : dev data ratio, default=0.10

    Speaker-independent 8:1:1 split.
    Returns {'train': [...], 'dev': [...], 'test': [...]}.
    """
    speakers, spk_to_recs = _speaker_shuffle(records)
    n = len(speakers)
    n_train = int(n * train_split)
    n_dev   = int(n * dev_split)

    train = [r for s in speakers[:n_train]           for r in spk_to_recs[s]]
    dev   = [r for s in speakers[n_train:n_train+n_dev] for r in spk_to_recs[s]]
    test  = [r for s in speakers[n_train+n_dev:]     for r in spk_to_recs[s]]
    return {'train': train, 'dev': dev, 'test': test}


def split_partial_carve_dev(train_records: list, 
                            test_records: list, 
                            train_split: float=0.80,
                            dev_split: float=0.10) -> dict:
    """
    Partial predetermined split (train+test only, no dev).
    Carve dev from train so overall ratio ≈ 8:1:1.
    Train-pool is split at 89:11 (= 0.8/0.9 : 0.1/0.9) speaker-independently.
    """
    speakers, spk_to_recs = _speaker_shuffle(train_records)
    n = len(speakers)
    # target: new_train : dev = 0.7 : 0.2 of total; test fixed
    # carve 22% of train speakers as dev
    n_new_train = int(n * (train_split / 0.9))
    new_train = [r for s in speakers[:n_new_train]   for r in spk_to_recs[s]]
    dev       = [r for s in speakers[n_new_train:]   for r in spk_to_recs[s]]
    return {'train': new_train, 'dev': dev, 'test': test_records}


def assign_by_membership(all_records: list,
                          membership: dict) -> dict:
    """
    Assign records to splits based on source_stem membership dict.
    membership: {source_stem: 'train' | 'dev' | 'test'}
    Records whose source_stem is not in membership are dropped with a warning.
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
        print(f"  WARN: {unassigned} records unassigned (source_stem not in membership)")
    return result


def save_manifest(key: str, lang: str,
                  split_source: str, split_note: str,
                  split_records: dict) -> None:
    """
    Save manifest JSON with split metadata.
    split_records: {'train': [...], 'dev': [...], 'test': [...]}
    Logs split source and actual ratio.
    """
    total  = sum(len(v) for v in split_records.values())
    counts = {s: len(v) for s, v in split_records.items()}
    ratio  = {s: round(c / max(total, 1), 3) for s, c in counts.items()}

    manifest = {
        "dataset_key":  key,
        "lang":         lang,
        "split_source": split_source,
        "split_note":   split_note,
        "total_utts":   total,
        "counts":       counts,
        "actual_ratio": ratio,
        "train":        split_records.get('train', []),
        "dev":          split_records.get('dev',   []),
        "test":         split_records.get('test',  []),
    }

    out_path = os.path.join(MANIFEST_DIR, f"{key}_manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Console log ──
    ratio_str = " | ".join(
        f"{s}={counts[s]:,} ({ratio[s]*100:.1f}%)" for s in ['train', 'dev', 'test']
    )
    split_tag = {
        'predetermined_full':           '✓ PREDETERMINED (full)',
        'predetermined_partial_carved': '⚠ PREDETERMINED (partial→carved dev)',
        '8_1_1':                        '→ SPLIT 8:2:1 (no original splits)',
    }.get(split_source, split_source)

    print(f"\n  [{key}] {split_tag}")
    print(f"  {split_note}")
    print(f"  Total {total:,} utts  |  {ratio_str}")


# ─── SEGMENTED DATASET PROCESSORS (from v1) ──────────────────────────────────

def _process_tsv_segmented(audio_dir: str, seg_map: dict,
                            lang: str, name: str,
                            output_dir: str) -> list:
    """Extract segments from long audio files using (start, end, text) tuples."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    skipped = 0

    audio_files = {}
    for ext in ['.mp3', '.wav', '.flac']:
        for af in Path(audio_dir).rglob(f'*{ext}'):
            audio_files[af.stem] = str(af)

    for audio_stem, segments in tqdm(seg_map.items(), desc=f"Segments {name}"):
        if audio_stem not in audio_files:
            skipped += len(segments)
            continue
        src = audio_files[audio_stem]

        for idx, (start, end, text) in enumerate(segments):
            dur = end - start
            if not (CONFIG["min_duration_sec"] <= dur <= CONFIG["max_duration_sec"]):
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue

            utt_id  = f"{lang}_{name}_{audio_stem}_{idx:04d}".replace(" ", "_")
            out_wav = os.path.join(output_dir, f"{utt_id}.wav")

            if not os.path.exists(out_wav):
                cmd = ["ffmpeg", "-y", "-i", src,
                       "-ss", str(start), "-t", str(dur),
                       "-ar", str(CONFIG["sample_rate"]),
                       "-ac", "1", "-acodec", "pcm_s16le",
                       out_wav, "-loglevel", "error"]
                if subprocess.run(cmd, capture_output=True).returncode != 0:
                    skipped += 1
                    continue

            actual = get_duration(out_wav)
            if actual < CONFIG["min_duration_sec"]:
                os.remove(out_wav)
                skipped += 1
                continue

            records.append({
                "utt_id":      utt_id,
                "wav_path":    out_wav,
                "text":        text.strip(),
                "speaker":     f"spk_{name}_{audio_stem[:8]}",
                "duration":    round(actual, 3),
                "source_stem": audio_stem,
            })

    print(f"  {name}: {len(records)} segs extracted, {skipped} skipped")
    return records


def _process_escwa_segmented(wav_dir: str, text_map: dict,
                              seg_map: dict, output_dir: str) -> list:
    """Extract ESCWA utterances using Kaldi segments timing."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    skipped = 0

    wav_files = {wf.stem: str(wf)
                 for wf in Path(wav_dir).rglob('*.wav')}

    for utt_id, (rec_id, start, end) in tqdm(seg_map.items(), desc="ESCWA"):
        if utt_id not in text_map:
            skipped += 1
            continue
        src = wav_files.get(rec_id) or next(
            (v for k, v in wav_files.items() if rec_id in k or k in rec_id), None)
        if not src:
            skipped += 1
            continue

        dur = end - start
        if not (CONFIG["min_duration_sec"] <= dur <= CONFIG["max_duration_sec"]):
            skipped += 1
            continue

        safe_id = utt_id.replace('/', '_')
        out_wav = os.path.join(output_dir, f"{safe_id}.wav")

        if not os.path.exists(out_wav):
            cmd = ["ffmpeg", "-y", "-i", src,
                   "-ss", str(start), "-t", str(dur),
                   "-ar", str(CONFIG["sample_rate"]),
                   "-ac", "1", "-acodec", "pcm_s16le",
                   out_wav, "-loglevel", "error"]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                skipped += 1
                continue

        actual = get_duration(out_wav)
        if actual < CONFIG["min_duration_sec"]:
            os.remove(out_wav)
            skipped += 1
            continue

        records.append({
            "utt_id":      safe_id,
            "wav_path":    out_wav,
            "text":        text_map[utt_id].strip(),
            "speaker":     f"spk_escwa_{rec_id[:8]}",
            "duration":    round(actual, 3),
            "source_stem": utt_id,
        })

    print(f"  ESCWA: {len(records)} segs, {skipped} skipped")
    return records


# ─── INDONESIAN DATASETS ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("INDONESIAN")
print("="*60)

# ── id_cv : PREDETERMINED FULL ───────────────────────────────────────────────
print("\n[id_cv] Mozilla Common Voice ID v24.0")
cv_id_root  = os.path.join(BASE_DATASET, "id", "mozilla", "scripted-id",
                            "cv-corpus-24.0-2025-12-05", "id")
cv_id_clips = os.path.join(cv_id_root, "clips")

split_records_cv_id = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(cv_id_root, f"{split}.tsv")
    if os.path.exists(tsv):
        m = load_mozilla_cv(tsv)
        recs = process_dataset(cv_id_clips,
                               os.path.join(BASE_OUT, "id", "cv", split),
                               "id", f"cv_{split}", m)
        split_records_cv_id[split] = recs
    else:
        print(f"  WARN: {tsv} not found — split '{split}' empty")
        split_records_cv_id[split] = []

save_manifest('id_cv', 'id', 'predetermined_full',
              'Mozilla CV ID v24.0 — TSV train/dev/test preserved',
              split_records_cv_id)

# ── id_fleurs : PREDETERMINED FULL ───────────────────────────────────────────
print("\n[id_fleurs] FLEURS ID")
fleurs_id_root = os.path.join(BASE_DATASET, "id", "Fleurs_id")

split_records_fleurs_id = {}
for split in ['train', 'dev', 'test']:
    tsv       = os.path.join(fleurs_id_root, f"{split}.tsv")
    audio_dir = os.path.join(fleurs_id_root, "audio", split)
    if os.path.exists(tsv) and os.path.exists(audio_dir):
        m = load_fleurs(tsv)
        recs = process_dataset(audio_dir,
                               os.path.join(BASE_OUT, "id", "fleurs", split),
                               "id", f"fleurs_{split}", m)
        split_records_fleurs_id[split] = recs
    else:
        print(f"  WARN: FLEURS ID split '{split}' — tsv or audio dir missing")
        split_records_fleurs_id[split] = []

save_manifest('id_fleurs', 'id', 'predetermined_full',
              'FLEURS ID — audio/train|dev|test dirs + TSV preserved',
              split_records_fleurs_id)

# ── id_librivox : PREDETERMINED PARTIAL ──────────────────────────────────────
print("\n[id_librivox] Librivox ID")
librivox_root = os.path.join(BASE_DATASET, "id", "Librivox")
train_meta    = os.path.join(librivox_root, "metadata_train.csv")
test_meta     = os.path.join(librivox_root, "metadata_test.csv")
train_audio   = os.path.join(librivox_root, "audio_train",
                              "librivox-indonesia", "train")
test_audio    = os.path.join(librivox_root, "audio_test",
                              "librivox-indonesia", "test")

librivox_train = process_dataset(
    train_audio, os.path.join(BASE_OUT, "id", "librivox", "train"),
    "id", "librivox_train", load_librivox_id(train_meta))
librivox_test  = process_dataset(
    test_audio, os.path.join(BASE_OUT, "id", "librivox", "test"),
    "id", "librivox_test", load_librivox_id(test_meta))

save_manifest('id_librivox', 'id', 'predetermined_partial_carved',
              'Librivox ID — metadata_train/test CSV; dev carved from train (78:22) for ≈7:2:1',
              split_partial_carve_dev(librivox_train, librivox_test))

# ── id_titml : 8:1:1 ─────────────────────────────────────────────────────────
print("\n[id_titml] TITML-IDN")
titml_dir = os.path.join(BASE_DATASET, "id", "TITML-IDN")
titml_map = load_titml_idn(titml_dir)
titml_all = process_dataset(titml_dir,
                             os.path.join(BASE_OUT, "id", "titml"),
                             "id", "titml", titml_map)

save_manifest('id_titml', 'id', '8_1_1',
              'TITML-IDN — speaker dirs only; speaker-independent 8:1:1 applied',
              split_data(titml_all))

# ── id_indocsc : 8:1:1 ───────────────────────────────────────────────────────
print("\n[id_indocsc] SEACrowd IndoCSC")
indocsc_wav = os.path.join(BASE_DATASET, "id", "SEACrowd",
                            "Indonesian_Conversational_Speech_Corpus", "WAV")
indocsc_txt = os.path.join(BASE_DATASET, "id", "SEACrowd",
                            "Indonesian_Conversational_Speech_Corpus", "TXT")
indocsc_all = process_dataset(indocsc_wav,
                               os.path.join(BASE_OUT, "id", "indocsc"),
                               "id", "indocsc",
                               load_seacrowd_indocsc(indocsc_wav, indocsc_txt))

save_manifest('id_indocsc', 'id', '8_1_1',
              'SEACrowd IndoCSC — no predetermined splits; 8:1:1 applied',
              split_data(indocsc_all))

# ── id_sindodsc : 8:1:1 ──────────────────────────────────────────────────────
print("\n[id_sindodsc] SEACrowd SIndoDuSC")
sindodsc_wav = os.path.join(BASE_DATASET, "id", "SEACrowd",
                             "Indonesian_Scripted_Speech_Corpus_Daily_Use_Sentence", "WAV")
sindodsc_all = process_dataset(sindodsc_wav,
                                os.path.join(BASE_OUT, "id", "sindodsc"),
                                "id", "sindodsc",
                                load_seacrowd_sindodsc(sindodsc_wav))

save_manifest('id_sindodsc', 'id', '8_1_1',
              'SEACrowd SIndoDuSC — WAV only; 8:1:1 applied (no transcript source confirmed)',
              split_data(sindodsc_all))

# ─── ARABIC DATASETS ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ARABIC")
print("="*60)

# ── ar_cv : PREDETERMINED FULL ───────────────────────────────────────────────
print("\n[ar_cv] Mozilla Common Voice AR v24.0")
cv_ar_root  = os.path.join(BASE_DATASET, "ar", "mozilla",
                            "cv-corpus-24.0-2025-12-05")
cv_ar_clips = os.path.join(cv_ar_root, "clips")

split_records_cv_ar = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(cv_ar_root, f"{split}.tsv")
    if os.path.exists(tsv):
        m = load_mozilla_cv(tsv)
        recs = process_dataset(cv_ar_clips,
                               os.path.join(BASE_OUT, "ar", "cv", split),
                               "ar", f"cv_{split}", m)
        split_records_cv_ar[split] = recs
    else:
        print(f"  WARN: {tsv} not found")
        split_records_cv_ar[split] = []

save_manifest('ar_cv', 'ar', 'predetermined_full',
              'Mozilla CV AR v24.0 — TSV train/dev/test preserved',
              split_records_cv_ar)

# ── ar_fleurs : PREDETERMINED FULL ───────────────────────────────────────────
print("\n[ar_fleurs] FLEURS AR")
fleurs_ar_root = os.path.join(BASE_DATASET, "ar", "fleurs-ar")

split_records_fleurs_ar = {}
for split in ['train', 'dev', 'test']:
    tsv       = os.path.join(fleurs_ar_root, f"{split}.tsv")
    audio_dir = os.path.join(fleurs_ar_root, "audio", split)
    if os.path.exists(tsv) and os.path.exists(audio_dir):
        m = load_fleurs(tsv)
        recs = process_dataset(audio_dir,
                               os.path.join(BASE_OUT, "ar", "fleurs", split),
                               "ar", f"fleurs_{split}", m)
        split_records_fleurs_ar[split] = recs
    else:
        print(f"  WARN: FLEURS AR split '{split}' missing")
        split_records_fleurs_ar[split] = []

save_manifest('ar_fleurs', 'ar', 'predetermined_full',
              'FLEURS AR — audio/train|dev|test dirs preserved',
              split_records_fleurs_ar)

# ── ar_clartts : PREDETERMINED PARTIAL ───────────────────────────────────────
print("\n[ar_clartts] ClArTTS")
clartts_dir = os.path.join(BASE_DATASET, "ar", "ClArTTS")

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

save_manifest('ar_clartts', 'ar', 'predetermined_partial_carved',
              'ClArTTS — train-*.parquet + test-*.parquet; dev carved from train for ≈7:2:1',
              split_partial_carve_dev(clartts_train_all, clartts_test_all))

# ─── ENGLISH DATASETS ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ENGLISH")
print("="*60)

# ── en_librispeech : 8:1:1 ───────────────────────────────────────────────────
print("\n[en_librispeech] LibriSpeech clean-100 (HF parquet)")
libri_dir = os.path.join(BASE_DATASET, "en", "librispeech", "Data")
libri_map = load_librispeech_parquet(libri_dir)
libri_extracted = os.path.join(libri_dir, "extracted")
libri_all = process_dataset(libri_extracted,
                             os.path.join(BASE_OUT, "en", "librispeech"),
                             "en", "librispeech", libri_map)

save_manifest('en_librispeech', 'en', '8_1_1',
              'LibriSpeech clean-100 HF parquet — all from train-clean-100; 8:1:1 applied',
              split_data(libri_all))

# ── en_fleurs : PREDETERMINED FULL ───────────────────────────────────────────
print("\n[en_fleurs] FLEURS EN")
fleurs_en_root = os.path.join(BASE_DATASET, "en", "fleurs_en")

split_records_fleurs_en = {}
for split in ['train', 'dev', 'test']:
    tsv       = os.path.join(fleurs_en_root, f"{split}.tsv")
    audio_dir = os.path.join(fleurs_en_root, "audio", split)
    if os.path.exists(tsv) and os.path.exists(audio_dir):
        m = load_fleurs(tsv)
        recs = process_dataset(audio_dir,
                               os.path.join(BASE_OUT, "en", "fleurs", split),
                               "en", f"fleurs_{split}", m)
        split_records_fleurs_en[split] = recs
    else:
        print(f"  WARN: FLEURS EN split '{split}' missing")
        split_records_fleurs_en[split] = []

save_manifest('en_fleurs', 'en', 'predetermined_full',
              'FLEURS EN — audio/train|dev|test dirs preserved',
              split_records_fleurs_en)

# ── en_cv_spon : 8:1:1 ───────────────────────────────────────────────────────
print("\n[en_cv_spon] Mozilla CV EN Spontaneous")
cv_en_dir = os.path.join(BASE_DATASET, "en", "mozilla",
                          "sps-corpus-1.0-2025-11-25-en")
cv_en_map = {}
for tsv_file in Path(cv_en_dir).glob('*.tsv'):
    cv_en_map.update(load_mozilla_cv(str(tsv_file)))
cv_en_all = process_dataset(os.path.join(cv_en_dir, "audios"),
                             os.path.join(BASE_OUT, "en", "cv_spon"),
                             "en", "cv_spon", cv_en_map)

save_manifest('en_cv_spon', 'en', '8_1_1',
              'Mozilla CV EN Spontaneous v1.0 — single corpus; 8:1:1 applied',
              split_data(cv_en_all))

# ─── CODE-SWITCHING DATASETS ─────────────────────────────────────────────────

print("\n" + "="*60)
print("CODE-SWITCHING")
print("="*60)

# ── cs_escwa : 8:1:1 ─────────────────────────────────────────────────────────
print("\n[cs_escwa] QCRI ESCWA")
escwa_dir = os.path.join(BASE_DATASET, "CS", "ar-en", "escwa", "cs.released")
escwa_text_map = load_escwa(escwa_dir)
escwa_seg_map  = load_escwa_segments(escwa_dir)
escwa_all = _process_escwa_segmented(
    os.path.join(escwa_dir, "wav"),
    escwa_text_map, escwa_seg_map,
    os.path.join(BASE_OUT, "cs", "escwa"))

save_manifest('cs_escwa', 'cs', '8_1_1',
              'QCRI ESCWA — Kaldi segments+text; no predetermined splits; 8:1:1 applied',
              split_data(escwa_all))

# ── cs_hari : 8:1:1 ──────────────────────────────────────────────────────────
print("\n[cs_hari] Hari Minggoean")
hari_dir     = os.path.join(BASE_DATASET, "CS", "id-en", "Hari Minggoean", "2")
hari_seg_map = load_hari_minggoean(hari_dir)
hari_all     = _process_tsv_segmented(hari_dir, hari_seg_map, "cs", "hari",
                                       os.path.join(BASE_OUT, "cs", "hari_minggoean"))

save_manifest('cs_hari', 'cs', '8_1_1',
              'Hari Minggoean — TSV segments (Audio file/Start/End/Text); 8:1:1 applied',
              split_data(hari_all))

# ── cs_homostoria : 8:1:1 ────────────────────────────────────────────────────
print("\n[cs_homostoria] Homostoria")
homo_dir     = os.path.join(BASE_DATASET, "CS", "id-en", "homostoria", "homostoria")
homo_seg_map = load_homostoria(homo_dir)
homo_all     = _process_tsv_segmented(homo_dir, homo_seg_map, "cs", "homostoria",
                                       os.path.join(BASE_OUT, "cs", "homostoria"))

save_manifest('cs_homostoria', 'cs', '8_1_1',
              'Homostoria — TSV segments; 8:1:1 applied',
              split_data(homo_all))

print("\n" + "="*60)
print(f"All manifests saved to: {MANIFEST_DIR}")
print("Next: run local/manifests_to_kaldi.py")
print("="*60)

# Run via windows/powershell:
# python run_preprocessing.py 2>&1 | Tee-Object -FilePath log/preprocessing_log.txt

# Run via linux:
# python run_preprocessing.py 2>&1 | tee log/preprocessing_log.txt