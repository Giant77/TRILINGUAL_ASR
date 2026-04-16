"""
run_preprocessing.py
Orchestrates all dataset preprocessing.
Outputs processed WAVs + a manifest JSON per language.
NEW: Manifest JSON used for Kaldi script generation in Phase 2
"""
import json
import os
from pathlib import Path
from preprocess_audio import (
    process_dataset, _process_tsv_segmented,
    _process_escwa_segmented
)  
from load_transcripts import (
    load_mozilla_cv, load_fleurs, load_titml_idn,
    load_librivox_id, load_seacrowd_indocsc,
    load_seacrowd_sindodsc, load_clartts,
    load_escwa, load_hari_minggoean, load_homostoria, 
    load_librispeech_parquet, load_escwa_segments,
    _process_escwa_segmented, load_mozilla_spontant
)

BASE_DATASET = r"D:\FYP\Trilingual_ASR\Dataset"  # CHANGE THIS
BASE_OUT     = r"D:\FYP\Trilingual_ASR\Dataset\processed"  # CHANGE THIS
# NEW: Record all paths for reproducibility log
MANIFEST_DIR = os.path.join(BASE_OUT, "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)

all_records = {}

# ─── INDONESIAN ──────────────────────────────────────────────────────────────

# 1. Mozilla CV ID
print("[ID] Mozilla Common Voice")
cv_id_dir   = os.path.join(BASE_DATASET, "id", "mozilla", "scripted-id",
                            "cv-corpus-24.0-2025-12-05", "id")
cv_id_clips = os.path.join(cv_id_dir, "clips")
cv_id_train = load_mozilla_cv(os.path.join(cv_id_dir, "train.tsv"))
cv_id_dev   = load_mozilla_cv(os.path.join(cv_id_dir, "dev.tsv"))
cv_id_test  = load_mozilla_cv(os.path.join(cv_id_dir, "test.tsv"))
cv_id_map   = {**cv_id_train, **cv_id_dev, **cv_id_test}
all_records['id_cv'] = process_dataset(cv_id_clips,
    os.path.join(BASE_OUT, "id", "cv"), "id", "cv", cv_id_map)

# 2. FLEURS ID
print("[ID] FLEURS")
fleurs_id_dir = os.path.join(BASE_DATASET, "id", "Fleurs_id")
fleurs_id_map = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(fleurs_id_dir, f"{split}.tsv")
    if os.path.exists(tsv):
        fleurs_id_map.update(load_fleurs(tsv))
all_records['id_fleurs'] = process_dataset(
    os.path.join(fleurs_id_dir, "audio"),
    os.path.join(BASE_OUT, "id", "fleurs"), "id", "fleurs", fleurs_id_map)

# 3. Librivox ID
print("[ID] Librivox")
librivox_meta_train = os.path.join(BASE_DATASET, "id", "Librivox", "metadata_train.csv", "metadata_train.csv")
librivox_meta_test  = os.path.join(BASE_DATASET, "id", "Librivox", "metadata_test.csv", "metadata_test.csv")
librivox_map = {}
librivox_map.update(load_librivox_id(librivox_meta_train))
librivox_map.update(load_librivox_id(librivox_meta_test))
librivox_audio_train = os.path.join(BASE_DATASET, "id", "Librivox",
                                     "audio_train", "librivox-indonesia", "train")
librivox_audio_test  = os.path.join(BASE_DATASET, "id", "Librivox",
                                     "audio_test", "librivox-indonesia", "test")
librivox_recs = []
librivox_recs += process_dataset(librivox_audio_train,
    os.path.join(BASE_OUT, "id", "librivox"), "id", "librivox", librivox_map)
librivox_recs += process_dataset(librivox_audio_test,
    os.path.join(BASE_OUT, "id", "librivox"), "id", "librivox", librivox_map)
all_records['id_librivox'] = librivox_recs

# 4. TITML-IDN
print("[ID] TITML-IDN")
titml_dir = os.path.join(BASE_DATASET, "id", "TITML-IDN")
titml_map = load_titml_idn(titml_dir)
all_records['id_titml'] = process_dataset(
    titml_dir, os.path.join(BASE_OUT, "id", "titml"),
    "id", "titml", titml_map)

# 5. SEACrowd IndoCSC
print("[ID] SEACrowd IndoCSC")
indocsc_wav = os.path.join(BASE_DATASET, "id", "SEACrowd",
                            "Indonesian_Conversational_Speech_Corpus", "WAV")
indocsc_txt = os.path.join(BASE_DATASET, "id", "SEACrowd",
                            "Indonesian_Conversational_Speech_Corpus", "TXT")
indocsc_map = load_seacrowd_indocsc(indocsc_wav, indocsc_txt)
all_records['id_indocsc'] = process_dataset(
    indocsc_wav, os.path.join(BASE_OUT, "id", "indocsc"),
    "id", "indocsc", indocsc_map)

# 6. SEACrowd SIndoDuSC
print("[ID] SEACrowd SIndoDuSC")
sindodsc_root = os.path.join(BASE_DATASET, "id", "SEACrowd",
                              "Indonesian_Scripted_Speech_Corpus_Daily_Use_Sentence")
sindodsc_wav  = os.path.join(sindodsc_root, "WAV")
sindodsc_map  = load_seacrowd_sindodsc(sindodsc_root)
# NEW: pass dataset_root (contains UTTRANSINFO.txt), wav search under WAV/
all_records['id_sindodsc'] = process_dataset(
    sindodsc_wav, os.path.join(BASE_OUT, "id", "sindodsc"),
    "id", "sindodsc", sindodsc_map)

# ─── ARABIC ──────────────────────────────────────────────────────────────────

# 7. Mozilla CV AR
print("[AR] Mozilla Common Voice")
cv_ar_dir   = os.path.join(BASE_DATASET, "ar", "mozilla",
                            "cv-corpus-24.0-2025-12-05", "clips")
cv_ar_meta  = os.path.join(BASE_DATASET, "ar", "mozilla",
                            "cv-corpus-24.0-2025-12-05")
cv_ar_map = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(cv_ar_meta, f"{split}.tsv")
    if os.path.exists(tsv):
        cv_ar_map.update(load_mozilla_cv(tsv))
all_records['ar_cv'] = process_dataset(
    cv_ar_dir, os.path.join(BASE_OUT, "ar", "cv"),
    "ar", "cv", cv_ar_map)


# 8. FLEURS AR
print("[AR] FLEURS")
fleurs_ar_dir = os.path.join(BASE_DATASET, "ar", "fleurs-ar")
fleurs_ar_map = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(fleurs_ar_dir, f"{split}.tsv")
    if os.path.exists(tsv):
        fleurs_ar_map.update(load_fleurs(tsv))
all_records['ar_fleurs'] = process_dataset(
    os.path.join(fleurs_ar_dir, "audio"),
    os.path.join(BASE_OUT, "ar", "fleurs"),
    "ar", "fleurs", fleurs_ar_map)

# 9. ClArTTS — parquet with audio bytes
print("[AR] ClArTTS")
clartts_dir = os.path.join(BASE_DATASET, "ar", "ClArTTS")
clartts_map = load_clartts(clartts_dir)
clartts_extracted = os.path.join(clartts_dir, "extracted")

all_records['ar_clartts'] = process_dataset(
    clartts_extracted,
    os.path.join(BASE_OUT, "ar", "clartts"),
    "ar", "clartts", clartts_map)

# ─── ENGLISH ─────────────────────────────────────────────────────────────────

# 10. LibriSpeech
print("[EN] LibriSpeech")
# CHANGES V1 — replace LibriSpeech section (was: .trans.txt flat file loader)
# changes the following on [EN] LibriSpeech section to:

# 10. LibriSpeech — HuggingFace parquet
print("[EN] LibriSpeech")
libri_dir = os.path.join(BASE_DATASET, "en", "librispeech", "Data")
# CHANGES V1: load_librispeech_parquet extracts WAVs to Data/extracted/
libri_map = load_librispeech_parquet(libri_dir)
libri_extracted = os.path.join(libri_dir, "extracted")
all_records['en_librispeech'] = process_dataset(
    libri_extracted,
    os.path.join(BASE_OUT, "en", "librispeech"),
    "en", "librispeech", libri_map)

# 11. FLEURS EN
print("[EN] FLEURS")
fleurs_en_dir = os.path.join(BASE_DATASET, "en", "fleurs_en")
fleurs_en_map = {}
for split in ['train', 'dev', 'test']:
    tsv = os.path.join(fleurs_en_dir, f"{split}.tsv")
    if os.path.exists(tsv):
        fleurs_en_map.update(load_fleurs(tsv))
all_records['en_fleurs'] = process_dataset(
    os.path.join(fleurs_en_dir, "audio"),
    os.path.join(BASE_OUT, "en", "fleurs"),
    "en", "fleurs", fleurs_en_map)

# 12. CV spon EN
print("[EN] Mozilla Spontaneous")
cv_en_spon_dir = os.path.join(BASE_DATASET, "en", "mozilla", "sps-corpus-1.0-2025-11-25-en")

cv_en_spon_map = load_mozilla_spontant(cv_en_spon_dir)

all_records['en_cv_spon'] = process_dataset(
    os.path.join(cv_en_spon_dir, "audios"),
    os.path.join(BASE_OUT, "en", "cv_spon"),
    "en", "cv_spon", cv_en_spon_map
)

# ─── CODE-SWITCHING ───────────────────────────────────────────────────────────

# 13. ESCWA — Kaldi segments + text files
print("[CS] ESCWA")
escwa_dir = os.path.join(BASE_DATASET, "CS", "ar-en", "escwa", "cs.released")
escwa_map  = load_escwa(escwa_dir)
escwa_segs = load_escwa_segments(escwa_dir)
escwa_wav_dir = os.path.join(escwa_dir, "wav")

# CHANGES V1: ESCWA has pre-defined utterance IDs in segments/text files.
# Audio files in wav/ are full recordings; we must extract segments.
escwa_records = _process_escwa_segmented(
    escwa_wav_dir, escwa_map, escwa_segs,
    os.path.join(BASE_OUT, "cs", "escwa"))
all_records['cs_escwa'] = escwa_records


# CHANGES V1 — replace Hari Minggoean section
# changes the following on [CS] Hari Minggoean section to:

# 14. Hari Minggoean — TSV-segmented (Audio file / Start / End / Text)
print("[CS] Hari Minggoean")
hari_dir = os.path.join(BASE_DATASET, "CS", "id-en", "Hari Minggoean", "2")
# CHANGES V1: returns {audio_stem: [(start, end, text), ...]}
hari_seg_map = load_hari_minggoean(hari_dir)
hari_records = _process_tsv_segmented(
    hari_dir, hari_seg_map, "cs", "hari",
    os.path.join(BASE_OUT, "cs", "hari_minggoean"))
all_records['cs_hari'] = hari_records

# 15. Homostoria — same structure
print("[CS] Homostoria")
homo_dir = os.path.join(BASE_DATASET, "CS", "id-en", "homostoria", "homostoria")
homo_seg_map = load_homostoria(homo_dir)
homo_records = _process_tsv_segmented(
    homo_dir, homo_seg_map, "cs", "homostoria",
    os.path.join(BASE_OUT, "cs", "homostoria"))
all_records['cs_homostoria'] = homo_records

# ─── SAVE MANIFESTS ──────────────────────────────────────────────────────────

for key, records in all_records.items():
    out_path = os.path.join(MANIFEST_DIR, f"{key}_manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {out_path} ({len(records)} utterances)")

print("\nPreprocessing complete. Transfer processed/ to WSL next.")

# Run via windows:
# python run_preprocessing.py 2>&1 | Tee-Object -FilePath preprocessing_log.txt

# Run via linux:
# python run_preprocessing.py 2>&1 | tee preprocessing_log.txt
