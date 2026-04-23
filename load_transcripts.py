"""
load_transcripts.py  —  v1 (CHANGES V1)
Dataset-specific transcript parsers.

Changes from v0:
  - load_titml_idn         : adds HTK MLF (words.mlf~) + script~ (suffix ~) parsing
  - load_clartts           : multipart parquet, 'audio' bytes + 'transcription' column
  - load_escwa             : Kaldi-style 'segments' + 'text' files (not STM/TSV)
  - load_hari_minggoean    : TSV columns: 'Audio file', 'Start', 'End', 'Text'
                             returns {audio_stem: [(start, end, text), ...]} for segmented extraction
  - load_librispeech_parquet: HuggingFace parquet with 'audio' bytes + 'text' + 'id' columns
  - load_seacrowd_sindodsc : adds parent-dir CSV/JSON search fallback
NEW: All loaders return dicts unless noted; parquet loaders require pyarrow + pandas
"""
import csv
import io
import os
import re
from pathlib import Path


# ─── SHARED UTILITIES ────────────────────────────────────────────────────────

def _try_parquet_import():
    """Lazy import check for parquet dependencies."""
    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        return pq, sf
    except ImportError as e:
        raise ImportError(
            f"Parquet loader requires: pip install pyarrow soundfile pandas\n{e}"
        )


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


def load_fleurs(tsv_path: str) -> dict:
    """
    Load FLEURS TSV.
    Returns: {file_stem: raw_transcription}
    """
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
            stem = os.path.splitext(row['file_name'])[0]
            result[stem] = row.get('raw_transcription',
                                   row.get('transcription', ''))
    return result


def load_titml_idn(data_dir: str) -> dict:
    """
    Load TITML-IDN transcripts.  CHANGES V1
    Tries in order:
      1. words.mlf~ (HTK Master Label File — most complete)
      2. script~ (per-speaker text, format: "*/UTTID transcript")
      3. Fallback .lab files per utterance (not present in this dataset)
    Returns: {utt_id: transcript}  e.g. {'01-001': 'hai selamat pagi'}
    """
    result = {}

    for spk_dir in sorted(Path(data_dir).iterdir()):
        if not spk_dir.is_dir():
            continue

        # ── Strategy 1: words.mlf~ (HTK MLF format) ──────────────────────
        mlf_file = spk_dir / 'words.mlf~'
        if mlf_file.exists():
            with open(mlf_file, encoding='utf-8', errors='replace') as f:
                current_id = None
                words = []
                for line in f:
                    line = line.rstrip()
                    if line.startswith('#!MLF!#'):
                        continue
                    if line.startswith('"'):
                        # New utterance header: "*/01-001.lab"
                        if current_id and words:
                            result[current_id] = ' '.join(words).lower()
                        # Extract utt_id from path: */SPEAKER/01-001.lab
                        parts = line.strip('"').replace('\\', '/').split('/')
                        stem = os.path.splitext(parts[-1])[0]
                        current_id = stem
                        words = []
                    elif line == '.':
                        if current_id and words:
                            result[current_id] = ' '.join(words).lower()
                        current_id = None
                        words = []
                    elif line and current_id is not None:
                        words.append(line.strip())
                if current_id and words:
                    result[current_id] = ' '.join(words).lower()
            continue  # MLF found — skip other strategies for this speaker

        # ── Strategy 2: script~ file ──────────────────────────────────────
        # Format: "*/UTTID transcript text here"
        # CHANGES V1: explicitly handle ~ suffix (not caught by f.suffix == '')
        script_file = None
        for candidate in spk_dir.iterdir():
            # Match any file whose stem contains 'script' regardless of suffix
            if 'script' in candidate.stem.lower() and not candidate.is_dir():
                script_file = candidate
                break

        if script_file is not None:
            script_pattern = re.compile(r'^\*/?([0-9A-Za-z\-_]+)\s+(.+)$')
            with open(script_file, encoding='utf-8', errors='replace') as f:
                for line in f:
                    m = script_pattern.match(line.strip())
                    if m:
                        utt_id = m.group(1)
                        text   = m.group(2).strip().lower()
                        result[utt_id] = text
            continue

        # ── Strategy 3: per-utterance .lab files ─────────────────────────
        for lab in spk_dir.glob('*.lab'):
            with open(lab, encoding='utf-8', errors='replace') as f:
                result[lab.stem] = f.read().strip().lower()

    return result


def load_librivox_id(metadata_csv: str) -> dict:
    """
    Load Librivox Indonesia metadata CSV.
    Returns {audio_stem: sentence}.
    """
    result = {}
    with open(metadata_csv, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # normalize language field
            lang = (row.get('language') or '').strip().lower()
            if lang != 'ind':
                continue


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
    Returns {audio_stem: transcript}.
    """
    result = {}
    for txt_file in Path(txt_dir).glob('*.txt'):
        with open(txt_file, encoding='utf-8') as f:
            result[txt_file.stem] = f.read().strip()
    return result


def load_seacrowd_sindodsc(wav_dir: str, dataset_root: str) -> dict:
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

def load_clartts(data_dir: str, parquet_files: list = None) -> dict:
    """
    Load ClArTTS from multipart Parquet.  CHANGES V1
    Parquet columns: 'audio' (dict with 'bytes' key), 'transcription' (Arabic text)
    Writes extracted WAVs to data_dir/extracted/ and returns {utt_id: text}.
    NOTE: Returns special format {utt_id: text} where utt_id is auto-generated.
          Audio is written to data_dir/extracted/{utt_id}.wav for process_dataset.
    
    Args:
        data_dir: Directory containing parquet files or extracted/ subdirectory
        parquet_files: Optional list of specific parquet file paths to load.
                      If None, auto-discovers all .parquet files in data_dir
    
    Returns: {utt_id: transcript}
    """
    pq, sf = _try_parquet_import()
    import pyarrow.parquet as pq_mod

    result = {}
    out_dir = Path(data_dir) / 'extracted'
    out_dir.mkdir(exist_ok=True)

    # Use specified parquet files or auto-discover
    if parquet_files is not None:
        files_to_load = [Path(pf) for pf in parquet_files]
    else:
        files_to_load = sorted(Path(data_dir).glob('*.parquet'))
        if not files_to_load:
            # Handle nested Dataset/ subdir
            files_to_load = sorted(Path(data_dir).rglob('*.parquet'))

    if not files_to_load:
        print(f"  WARN [ClArTTS]: No parquet files in {data_dir}")
        return result

    global_idx = 0
    for pf in files_to_load:
        table = pq_mod.read_table(str(pf))
        rows = table.to_pylist()
        for row in rows:
            utt_id   = f"clartts_{global_idx:06d}"
            text     = row.get('transcription', row.get('text', ''))
            audio_v  = row.get('audio', {})

            if not text:
                global_idx += 1
                continue

            # Extract audio bytes
            if isinstance(audio_v, dict):
                audio_bytes = audio_v.get('bytes', b'')
            elif isinstance(audio_v, bytes):
                audio_bytes = audio_v
            else:
                global_idx += 1
                continue

            if not audio_bytes:
                global_idx += 1
                continue

            # Write WAV
            out_wav = out_dir / f"{utt_id}.wav"
            if not out_wav.exists():
                try:
                    with io.BytesIO(audio_bytes) as bio:
                        data, sr = sf.read(bio)
                    sf.write(str(out_wav), data, sr)
                except Exception as e:
                    print(f"  WARN [ClArTTS] write error {utt_id}: {e}")
                    global_idx += 1
                    continue

            result[utt_id] = text.strip()
            global_idx += 1

    print(f"  [ClArTTS] Extracted {len(result)} utterances to {out_dir}")
    return result


# ─── CODE-SWITCHING ───────────────────────────────────────────────────────────

def load_escwa(data_dir: str) -> dict:
    """
    Load QCRI/escwa CS dataset.  CHANGES V1
    Format: Kaldi-style 'segments' + 'text' files (NOT STM or TSV)

    segments format: utt_id  recording_id  start_sec  end_sec
    text format:     utt_id  transcription

    Returns: {utt_id: text}
    NOTE: Audio segmentation from long recordings is handled separately
          in run_preprocessing.py using the segment timestamps.
    """
    result = {}
    data_path = Path(data_dir)

    text_file     = data_path / 'text'
    segments_file = data_path / 'segments'

    if not text_file.exists():
        print(f"  WARN [ESCWA]: 'text' file not found in {data_dir}")
        return result

    # Load transcripts
    with open(text_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]

    print(f"  [ESCWA] Loaded {len(result)} transcripts from text file")

    # Segments info (returned separately if caller needs it)
    # Stored as result with special prefix for callers that need timing
    if segments_file.exists():
        seg_map = {}
        with open(segments_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    utt_id, rec_id, start, end = parts
                    seg_map[utt_id] = (rec_id, float(start), float(end))
        # Attach segment info to result as metadata (won't interfere with text)
        # Accessible via load_escwa_segments() below
        _ESCWA_SEGMENTS.clear()
        _ESCWA_SEGMENTS.update(seg_map)
        print(f"  [ESCWA] Loaded {len(seg_map)} segment timings")

    return result


_ESCWA_SEGMENTS: dict = {}  # {utt_id: (rec_id, start_sec, end_sec)}


def load_escwa_segments(data_dir: str) -> dict:
    """
    Returns ESCWA segment timing: {utt_id: (recording_id, start_sec, end_sec)}.
    Call after load_escwa() to get timing info for audio extraction.
    """
    if not _ESCWA_SEGMENTS:
        segments_file = Path(data_dir) / 'segments'
        if segments_file.exists():
            with open(segments_file, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        utt_id, rec_id, start, end = parts
                        _ESCWA_SEGMENTS[utt_id] = (rec_id, float(start), float(end))
    return dict(_ESCWA_SEGMENTS)

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

# ─── ENGLISH ─────────────────────────────────────────────────────────────────

def load_librispeech_parquet(data_dir: str) -> dict:
    """
    Load LibriSpeech from HuggingFace parquet.  NEW in CHANGES V1
    Parquet columns: 'file'(unused), 'audio'(bytes), 'text'(uppercase),
                     'speaker_id', 'chapter_id', 'id'
    Writes extracted WAVs to data_dir/extracted/ and returns {utt_id: text}.
    'id' column used as utt_id (e.g., '1272-128104-0000').
    Returns: {utt_id: text_lowercase}
    """
    pq, sf = _try_parquet_import()
    import pyarrow.parquet as pq_mod

    result = {}
    out_dir = Path(data_dir) / 'extracted'
    out_dir.mkdir(exist_ok=True)

    parquet_files = sorted(Path(data_dir).glob('*.parquet'))
    if not parquet_files:
        parquet_files = sorted(Path(data_dir).rglob('*.parquet'))

    if not parquet_files:
        print(f"  WARN [LibriSpeech]: No parquet files in {data_dir}")
        return result

    for pf in parquet_files:
        table = pq_mod.read_table(str(pf))
        rows  = table.to_pylist()
        for row in rows:
            utt_id    = str(row.get('id', '')).strip()
            text      = row.get('text', '').strip().lower()  # lowercase per convention
            audio_v   = row.get('audio', {})
            speaker_id = str(row.get('speaker_id', 'unk'))

            if not utt_id or not text:
                continue

            if isinstance(audio_v, dict):
                audio_bytes = audio_v.get('bytes', b'')
            elif isinstance(audio_v, bytes):
                audio_bytes = audio_v
            else:
                continue

            if not audio_bytes:
                continue

            out_wav = out_dir / f"{utt_id}.wav"
            if not out_wav.exists():
                try:
                    with io.BytesIO(audio_bytes) as bio:
                        data, sr = sf.read(bio)
                    sf.write(str(out_wav), data, sr)
                except Exception as e:
                    print(f"  WARN [LibriSpeech] {utt_id}: {e}")
                    continue

            result[utt_id] = text

    print(f"  [LibriSpeech] Extracted {len(result)} utterances to {out_dir}")
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

    - invalid TSV (optional, e.g., reorted.tsv)
      Columns:
        client_id, audio_id, audio_file, duration_ms, prompt_id,
        prompt, reason, comment, language

    Returns:
        {audio_filename_stem: transcription}
    """

    result = {}
    invalid_ids = set()

    # 1. Collect invalid samples (if exist)
    for tsv_file in Path(tsv_dir).glob('*reported*.tsv'):
        with open(tsv_file, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                stem = os.path.splitext(row['audio_file'])[0]
                invalid_ids.add(stem)

    # 2. Load valid samples
    for tsv_file in Path(tsv_dir).glob('*.tsv'):
        if 'reported' in tsv_file.name:
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
