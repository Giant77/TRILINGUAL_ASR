"""
load_transcripts.py
Dataset-specific transcript parsers for each source in the project.
NEW: Modular design — each parser is independent for easy debugging (TA constraint)
"""
import csv
import os
import re
from pathlib import Path


# ─── INDONESIAN ──────────────────────────────────────────────────────────────

def load_mozilla_cv(tsv_path: str) -> dict:
    """
    Load Mozilla Common Voice TSV.
    Columns: client_id, path, sentence, up_votes, down_votes, age, gender, accents
    Returns: {audio_filename_stem: sentence}
    """
    result = {}
    with open(tsv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # 'path' field may include .mp3 extension
            stem = os.path.splitext(row['path'])[0]
            result[stem] = row['sentence']
    return result

def load_mozilla_spontant(tsv_dir: str) -> dict:
    """
    Load Mozilla Spontaneous Speech dataset.

    Expected files inside tsv_dir:
    - main TSV (e.g., train.tsv / validated.tsv)
      Columns:
        client_id, audio_id, audio_file, duration_ms, prompt_id,
        prompt, transcription, votes, age, gender, language,
        quality_tags, split, char_per_sec

    - invalid TSV (optional, e.g., invalidated.tsv)
      Columns:
        client_id, audio_id, audio_file, duration_ms, prompt_id,
        prompt, reason, comment, language

    Returns:
        {audio_filename_stem: transcription}
    """

    result = {}
    invalid_ids = set()

    # 1. Collect invalid samples (if exist)
    for tsv_file in Path(tsv_dir).glob('*invalid*.tsv'):
        with open(tsv_file, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                stem = os.path.splitext(row['audio_file'])[0]
                invalid_ids.add(stem)

    # 2. Load valid samples
    for tsv_file in Path(tsv_dir).glob('*.tsv'):
        if 'invalid' in tsv_file.name:
            continue

        with open(tsv_file, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if not row.get('transcription'):
                    continue

                stem = os.path.splitext(row['audio_file'])[0]

                # skip invalid samples
                if stem in invalid_ids:
                    continue

                # optional: filter empty/noisy transcription
                text = row['transcription'].strip()
                if text == "":
                    continue

                result[stem] = text

    return result


def load_fleurs(tsv_path: str) -> dict:
    fieldnames = [
        "id",
        "file_name",
        "raw_transcription",
        "transcription",
        "phonemes",
        "num_samples",
        "gender",
    ]

    result = {}
    with open(tsv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        for row in reader:
            stem = os.path.splitext(row["file_name"])[0]
            result[stem] = row["raw_transcription"] or row["transcription"]
    return result


def load_titml_idn(data_dir: str) -> dict:
    """
    Load TITML-IDN transcripts.
    Format per file: "*/UTTERANCE-ID transcript text here"
    Speaker dirs: 01–20 under data_dir
    Returns: {utt_id: transcript}
    NEW: TITML-IDN uses speaker-numbered dirs; utt_id format = spkNN_uttMMM
    """
    result = {}
    script_pattern = re.compile(r'^\*/?([\w\-]+)\s+(.+)$')

    for spk_dir in sorted(Path(data_dir).iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name.zfill(2)

        # Find transcript files (files starting with "script" or similar)
        for f in spk_dir.iterdir():
            if 'script' in f.name.lower() and f.suffix in ['', '.txt', '.lab']:
                with open(f, encoding='utf-8', errors='replace') as fp:
                    for line in fp:
                        line = line.strip()
                        m = script_pattern.match(line)
                        if m:
                            utt_raw = m.group(1)
                            text = m.group(2).strip()
                            # Normalize utt_id to match expected audio filename
                            result[utt_raw] = text
            # Also check .lab files per utterance
            elif f.suffix == '.lab':
                with open(f, encoding='utf-8', errors='replace') as fp:
                    text = fp.read().strip()
                    result[f.stem] = text

    return result


def load_librivox_id(metadata_csv: str) -> dict:
    """
    Load Librivox Indonesia metadata CSV.
    Expected columns: id, path, sentence (or similar)
    Returns: {audio_stem: sentence}
    """
    result = {}
    with open(metadata_csv, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Adapt column names based on actual file
            path_col = row.get('path', row.get('file', row.get('filename', '')))
            text_col = row.get('sentence', row.get('transcription',
                                                    row.get('text', '')))
            if path_col and text_col:
                stem = os.path.splitext(os.path.basename(path_col))[0]
                result[stem] = text_col
    return result


def load_seacrowd_indocsc(wav_dir: str, txt_dir: str) -> dict:
    """
    Load SEACrowd ASR-IndoCSC.
    Assumes .wav files in wav_dir with matching .txt in txt_dir
    Returns: {audio_stem: transcript}
    """
    result = {}
    for txt_file in Path(txt_dir).glob('*.txt'):
        with open(txt_file, encoding='utf-8') as f:
            result[txt_file.stem] = f.read().strip()
    return result


def load_seacrowd_sindodsc(dataset_root: str) -> dict:
    """
    Load SEACrowd ASR-SIndoDuSC transcripts from UTTRANSINFO.txt.
    File location: <dataset_root>/UTTRANSINFO.txt
    Columns (tab-sep): CHANNEL  UTTRANS_ID  SPEAKER_ID  PROMPT  TRANSCRIPTION
    UTTRANS_ID is the wav filename, e.g. G0004_0_S0001.wav
    Returns: {wav_stem: transcription}
    NEW: UTTRANSINFO.txt confirmed as authoritative transcript source
    """
    result = {}
    info_path = os.path.join(dataset_root, 'UTTRANSINFO.txt')
    if not os.path.exists(info_path):
        print(f"  WARN: UTTRANSINFO.txt not found at {info_path}")
        return result

    with open(info_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uttrans_id  = row.get('UTTRANS_ID', '').strip()
            transcription = row.get('TRANSCRIPTION', '').strip()
            if uttrans_id and transcription:
                # UTTRANS_ID = "G0004_0_S0001.wav" → stem = "G0004_0_S0001"
                stem = os.path.splitext(uttrans_id)[0]
                result[stem] = transcription
    return result

# ─── ARABIC ──────────────────────────────────────────────────────────────────

def load_clartts(data_dir: str) -> dict:
    """
    Load ClArTTS dataset.
    WARN: Exact format unclear — see INFO REQUEST IR-2
    Assumed: metadata.csv or per-utterance .lab files
    """
    result = {}
    meta = Path(data_dir) / 'metadata.csv'
    if meta.exists():
        with open(meta, encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='|')  # LJSpeech-style
            for row in reader:
                if len(row) >= 2:
                    result[row[0]] = row[1]
        return result

    # Fallback: .lab files
    for lab_file in Path(data_dir).rglob('*.lab'):
        with open(lab_file, encoding='utf-8', errors='replace') as f:
            result[lab_file.stem] = f.read().strip()
    return result


# ─── CODE-SWITCHING ───────────────────────────────────────────────────────────

def load_escwa(data_dir: str) -> dict:
    """
    Load QCRI/escwa CS dataset.
    Expected: wav files + transcript file (stm, tsv, or txt)
    Returns: {audio_stem: transcript}
    WARN: Format verification needed — see INFO REQUEST IR-3
    """
    result = {}
    wav_dir = Path(data_dir) / 'wav'
    if not wav_dir.exists():
        wav_dir = Path(data_dir)

    # Try STM format
    for stm_file in Path(data_dir).rglob('*.stm'):
        with open(stm_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    utt_id = parts[0]
                    result[utt_id] = ' '.join(parts[5:])
        return result

    # Try TSV
    for tsv_file in Path(data_dir).rglob('*.tsv'):
        result.update(load_mozilla_cv(str(tsv_file)))
    return result


def _parse_hhmmss(ts: str) -> float:
    """Convert H:MM:SS or M:SS to seconds."""
    parts = ts.strip().split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def load_hari_minggoean(data_dir: str) -> list:
    """
    Load Hari Minggoean podcast CS dataset.
    TSV filename: "Transcript_Hari Minggoean.tsv"
    Columns (tab-sep): Audio file | Start | End | Text
    'Audio file' = stem without extension, e.g. "Hari Minggoean_001"
    Start/End = H:MM:SS timestamps within the corresponding mp3.
    Returns: list of dicts {audio_stem, start_sec, end_sec, text}
    NOTE: audio_stem must be matched to actual mp3 in directory (case-sensitive).
    NEW: Returns segment list, NOT stem→text dict — requires timestamp-aware
         processing in run_preprocessing.py (see process_podcast_segments)
    NEW: TSV confirmed columns: "Audio file", "Start", "End", "Text"
    """
    records = []
    tsv_path = None
    for f in Path(data_dir).rglob('*.tsv'):
        if 'transcript' in f.name.lower() or 'minggoean' in f.name.lower():
            tsv_path = f
            break
    if tsv_path is None:
        print(f"  WARN: No transcript TSV found in {data_dir}")
        return records

    with open(tsv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            audio_stem = row.get('Audio file', '').strip()
            start_str  = row.get('Start', '').strip()
            end_str    = row.get('End', '').strip()
            text       = row.get('Text', '').strip()
            if not audio_stem or not text:
                continue
            try:
                start_sec = _parse_hhmmss(start_str)
                end_sec   = _parse_hhmmss(end_str)
            except (ValueError, IndexError):
                print(f"  WARN: bad timestamp row {i}: {start_str}–{end_str}")
                continue
            if end_sec <= start_sec:
                continue
            records.append({
                'audio_stem': audio_stem,
                'start_sec':  round(start_sec, 3),
                'end_sec':    round(end_sec, 3),
                'text':       text
            })
    return records


def load_homostoria(data_dir: str) -> dict:
    """
    Load Homostoria podcast CS dataset.
    Attempts same TSV parsing as Hari Minggoean (timestamped segments).
    Returns segment list if TSV found; empty list otherwise.
    NEW: IR-7 — Homostoria transcript format unconfirmed; see info request
    """
    return load_hari_minggoean(data_dir)
