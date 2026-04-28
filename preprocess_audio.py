"""
preprocess_audio.py
Purpose: Convert all audio to 16kHz mono WAV, segment long files, skip short files
NEW: Undergraduate constraint — fixed seed for reproducibility (required for TA defense)
"""
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# NEW: Fixed configuration for reproducibility
CONFIG = {
    "sample_rate": 16000,
    "min_duration_sec": 1.0,
    "max_duration_sec": 30.0,
    "target_duration_sec": 10.0, # target for segmentation
    "silence_threshold_db": -50,
    "silence_min_len_ms": 500,
}


# ─── SEGMENTED DATASET PROCESSORS ────────────────────────────────────────────
def process_escwa_segmented(wav_dir: str, text_map: dict,
                               seg_map: dict, output_dir: str) -> list:
    """
    Process ESCWA: extract segments from long recordings using Kaldi segments file.
    seg_map: {utt_id: (rec_id, start_sec, end_sec)}
    text_map: {utt_id: transcript}
    CHANGES V1: uses Kaldi segments format for timing
    """
    import subprocess
    from pathlib import Path as _Path
    os.makedirs(output_dir, exist_ok=True)
    records = []
    skipped = 0

    wav_files = {}
    for wf in _Path(wav_dir).rglob('*.wav'):
        wav_files[wf.stem] = str(wf)

    for utt_id, (rec_id, start, end) in tqdm(seg_map.items(), desc="ESCWA"):
        if utt_id not in text_map:
            skipped += 1
            continue

        # Match recording ID to wav file (partial match)
        src_wav = wav_files.get(rec_id)
        if not src_wav:
            # Try prefix match
            matches = [v for k, v in wav_files.items() if rec_id in k or k in rec_id]
            src_wav = matches[0] if matches else None
        if not src_wav:
            skipped += 1
            continue

        duration = end - start
        out_wav = os.path.join(output_dir, f"{utt_id.replace('/', '_')}.wav")
        if not os.path.exists(out_wav):
            cmd = [
                "ffmpeg", "-y", "-i", src_wav,
                "-ss", str(start), "-t", str(duration),
                "-ar", str(CONFIG["sample_rate"]),
                "-ac", "1", "-acodec", "pcm_s16le",
                out_wav, "-loglevel", "error"
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                skipped += 1
                continue

        actual_dur = get_duration(out_wav)
        if actual_dur < CONFIG["min_duration_sec"]:
            _record_short_segment('cs', {
                "utt_id":   utt_id.replace('/', '_'),
                "wav_path": out_wav,
                "text":     text_map[utt_id].strip(),
                "speaker":  f"spk_escwa_{rec_id[:8]}",
                "duration": round(actual_dur, 3),
                "source_stem": utt_id,
                "source_dataset": 'cs_escwa',
            })
            skipped += 1
            continue

        if actual_dur > CONFIG["max_duration_sec"]:
            segments = segment_long_audio(out_wav, output_dir,
                                           utt_id, text_map[utt_id].strip())
            os.remove(out_wav)
            for seg_path, seg_text in segments:
                seg_id = os.path.splitext(os.path.basename(seg_path))[0]
                seg_dur = get_duration(seg_path)
                record = {
                    "utt_id":      seg_id,
                    "wav_path":    seg_path,
                    "text":        seg_text.strip(),
                    "speaker":     f"spk_escwa_{rec_id[:8]}",
                    "duration":    round(seg_dur, 3),
                    "source_stem": utt_id,
                    "source_dataset": 'cs_escwa',
                }
                if seg_dur < CONFIG["min_duration_sec"]:
                    _record_short_segment('cs', record)
                else:
                    records.append(record)
            continue

        records.append({
            "utt_id":   utt_id.replace('/', '_'),
            "wav_path": out_wav,
            "text":     text_map[utt_id].strip(),
            "speaker":  f"spk_escwa_{rec_id[:8]}",
            "duration": round(actual_dur, 3),
            "source_stem": utt_id,
            "source_dataset": 'cs_escwa',
        })

    print(f"  ESCWA: {len(records)} segments, {skipped} skipped")
    return records

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Convert any audio to 16kHz mono WAV using ffmpeg.

    Returns:
        bool: True if conversion succeeds, False otherwise.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(CONFIG["sample_rate"]),
        "-ac", "1",  # mono
        "-acodec", "pcm_s16le",
        output_path,
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    # result = subprocess.run(cmd)

    if result.returncode != 0:
        stderr_msg = result.stderr.decode().strip() if result.stderr else "No stderr output"
        print(f"Error: ffmpeg failed to convert {input_path} to {output_path}. stderr: {stderr_msg}")
        return False
    return True

def get_duration(wav_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

SHORT_SEGMENT_RECORDS = defaultdict(list)


def _record_short_segment(lang: str, record: dict) -> None:
    SHORT_SEGMENT_RECORDS[lang].append(record)


def save_short_segment_manifests(manifest_dir: str) -> None:
    os.makedirs(manifest_dir, exist_ok=True)
    for lang, records in SHORT_SEGMENT_RECORDS.items():
        total = len(records)
        hours = sum(r.get('duration', 0) for r in records) / 3600.0
        manifest = {
            'lang':          lang,
            'total_utts':    total,
            'total_hours':   round(hours, 2),
            'short_records': records,
        }
        out_path = os.path.join(manifest_dir, f'short_segments_{lang}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  Saved short-segments manifest for {lang}: {out_path} ({total} utts, {hours:.2f}h)")


def _split_transcript_by_durations(text: str, durations: list[float]) -> list[str]:
    words = text.strip().split()
    if not words:
        return [''] * len(durations)

    total = sum(durations)
    assert total > 0, 'Segment durations must be positive'

    target_counts = [max(1, round(len(words) * (dur / total))) for dur in durations]
    diff = len(words) - sum(target_counts)
    idx = 0
    while diff != 0:
        if diff > 0:
            target_counts[idx % len(target_counts)] += 1
            diff -= 1
        else:
            if target_counts[idx % len(target_counts)] > 1:
                target_counts[idx % len(target_counts)] -= 1
                diff += 1
        idx += 1

    output = []
    offset = 0
    for count in target_counts:
        chunk = words[offset:offset + count]
        output.append(' '.join(chunk))
        offset += count
    return output


def _build_fixed_split_durations(duration: float) -> list[float]:
    if duration <= CONFIG['max_duration_sec']:
        return [duration]

    import math
    num_chunks = max(2, math.ceil(duration / CONFIG['max_duration_sec']))
    chunk_len = duration / num_chunks
    if chunk_len < CONFIG['min_duration_sec']:
        num_chunks = math.ceil(duration / CONFIG['min_duration_sec'])
        chunk_len = duration / num_chunks

    durations = [round(chunk_len, 3)] * num_chunks
    total = sum(durations)
    if total != duration:
        durations[-1] += round(duration - total, 3)
    return durations


def segment_long_audio(input_path: str, output_dir: str,
                        utt_id: str, transcript: str) -> list:
    """
    Split audio > max_duration into segments using silence detection.
    If silence trimming still leaves segments > max_duration, split with transcript alignment.
    Returns list of (segment_path, segment_transcript) tuples.
    """
    duration = get_duration(input_path)
    if duration <= CONFIG['max_duration_sec']:
        return [(input_path, transcript)]

    segment_cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', f"silencedetect=noise={CONFIG['silence_threshold_db']}dB:d={CONFIG['silence_min_len_ms']/1000}",
        '-f', 'null', '-'
    ]
    result = subprocess.run(segment_cmd, capture_output=True, text=True)

    silence_starts = []
    silence_ends = []
    for line in result.stderr.split('\n'):
        if 'silence_start:' in line:
            try:
                silence_starts.append(float(line.split('silence_start:')[1].strip()))
            except (IndexError, ValueError):
                pass
        elif 'silence_end:' in line:
            try:
                silence_ends.append(float(line.split('silence_end:')[1].split(' ')[0].strip()))
            except (IndexError, ValueError):
                pass

    if silence_starts and silence_ends:
        split_points = [0.0]
        for s, e in zip(silence_starts, silence_ends):
            split_points.append((s + e) / 2.0)
        split_points.append(duration)
        candidate_ranges = [
            (split_points[i], split_points[i + 1])
            for i in range(len(split_points) - 1)
            if split_points[i + 1] - split_points[i] >= CONFIG['min_duration_sec']
        ]
    else:
        candidate_ranges = [(0.0, duration)]

    final_segments = []
    for idx, (start, end) in enumerate(candidate_ranges):
        seg_dur = end - start
        if seg_dur <= CONFIG['max_duration_sec']:
            final_segments.append((start, seg_dur))
            continue

        durations = _build_fixed_split_durations(seg_dur)
        current = start
        for d in durations:
            final_segments.append((current, d))
            current += d

    if not final_segments:
        return [(input_path, transcript)]

    durations = [seg_dur for _, seg_dur in final_segments]
    transcript_chunks = _split_transcript_by_durations(transcript, durations)

    segments = []
    current = 0
    for idx, ((start, seg_dur), seg_text) in enumerate(zip(final_segments, transcript_chunks)):
        seg_path = os.path.join(output_dir, f"{utt_id}_seg{idx:03d}.wav")
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start), '-t', str(seg_dur),
            '-ar', str(CONFIG['sample_rate']), '-ac', '1',
            '-acodec', 'pcm_s16le', seg_path, '-loglevel', 'error'
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(seg_path):
            actual_dur = get_duration(seg_path)
            if actual_dur >= CONFIG['min_duration_sec']:
                segments.append((seg_path, seg_text.strip()))
        current += seg_dur

    return segments if segments else [(input_path, transcript)]


def process_dataset(input_dir: str, output_dir: str,
                     lang: str, dataset_name: str,
                     transcript_map: dict) -> list:
    """
    Generic processing function.  
    transcript_map: {filename_without_ext: transcript_text}  
    Returns list of {utt_id, wav_path, text, duration} dicts  
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []
    skipped = 0

    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(Path(input_dir).rglob(f'*{ext}'))

    for audio_path in tqdm(audio_files, desc=f"Processing {dataset_name}"):
        stem = audio_path.stem
        if stem not in transcript_map:
            skipped += 1
            continue

        transcript = transcript_map[stem]
        if not transcript or len(transcript.strip()) < 2:
            skipped += 1
            continue

        # Sanitize utterance ID
        utt_id = f"{lang}_{dataset_name}_{stem}".replace(" ", "_")
        out_wav = os.path.join(output_dir, f"{utt_id}.wav")

        if not convert_to_wav(str(audio_path), out_wav):
            skipped += 1
            continue

        duration = get_duration(out_wav)
        if duration < CONFIG["min_duration_sec"]:
            record = {
                "utt_id":      utt_id,
                "wav_path":    out_wav,
                "text":        transcript.strip().lower() if lang != "ar" else transcript.strip(),
                "speaker":     f"spk_{dataset_name}_{stem[:6]}",
                "duration":    round(duration, 3),
                "source_stem": stem,
                "source_dataset": dataset_name,
            }
            _record_short_segment(lang, record)
            continue

        if duration > CONFIG["max_duration_sec"]:
            segments = segment_long_audio(out_wav, output_dir,
                                           utt_id, transcript)
            os.remove(out_wav)  # remove the unsegmented version
            for seg_path, seg_text in segments:
                seg_id = os.path.splitext(os.path.basename(seg_path))[0]
                seg_dur = get_duration(seg_path)
                record = {
                    "utt_id":      seg_id,
                    "wav_path":    seg_path,
                    "text":        seg_text.strip().lower() if lang != "ar" else seg_text.strip(),
                    "speaker":     f"spk_{dataset_name}_{stem[:6]}",
                    "duration":    round(seg_dur, 3),
                    "source_stem": stem,
                    "source_dataset": dataset_name,
                }
                if seg_dur < CONFIG["min_duration_sec"]:
                    _record_short_segment(lang, record)
                else:
                    records.append(record)
        else:
            records.append({
                "utt_id":      utt_id,
                "wav_path":    out_wav,
                "text":        transcript.strip().lower() if lang != "ar" else transcript.strip(),
                "speaker":     f"spk_{dataset_name}_{stem[:6]}",
                "duration":    round(duration, 3),
                "source_stem": stem,
                "source_dataset": dataset_name,
            })

    print(f"  {dataset_name}: {len(records)} processed, {skipped} skipped")
    return records


def process_podcast_segments(data_dir: str, output_dir: str,
                               lang: str, dataset_name: str,
                               segment_records: list) -> list:
    """
    Extract audio segments from podcast mp3/wav files using timestamps.
    segment_records: list of {audio_stem, start_sec, end_sec, text}
    Each audio_stem is matched to a file in data_dir (case-insensitive glob).
    Returns list of {utt_id, wav_path, text, speaker, duration} dicts.
    NEW: Handles long-form podcast audio (20min+) via timestamp-based cutting
    NEW: Case-insensitive stem matching handles "MInggoean" typo in filename
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    skipped = 0

    # Build case-insensitive stem → path index
    audio_index = {}
    for ext in ['.mp3', '.wav', '.flac']:
        for f in Path(data_dir).glob(f'*{ext}'):
            audio_index[f.stem.lower()] = str(f)

    for i, rec in enumerate(tqdm(segment_records,
                                   desc=f"Segments {dataset_name}")):
        audio_stem = rec['audio_stem']
        start_sec  = rec['start_sec']
        end_sec    = rec['end_sec']
        text       = rec['text'].strip()
        duration   = end_sec - start_sec

        # Extract segment with ffmpeg
        src_path = audio_index.get(audio_stem.lower())
        if src_path is None:
            # Try partial match (first word match for typos)
            for key, path in audio_index.items():
                if audio_stem.lower()[:10] in key:
                    src_path = path
                    break
        if src_path is None:
            skipped += 1
            continue

        safe_stem = audio_stem.replace(' ', '_').replace('/', '_')
        utt_id    = f"{lang}_{dataset_name}_{safe_stem}_s{i:05d}"
        out_wav   = os.path.join(output_dir, f"{utt_id}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ss", str(start_sec),
            "-t",  str(duration),
            "-ar", str(CONFIG["sample_rate"]),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            out_wav, "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not os.path.exists(out_wav):
            skipped += 1
            continue

        actual_dur = get_duration(out_wav)
        if actual_dur < CONFIG["min_duration_sec"]:
            _record_short_segment(lang, {
                "utt_id":      utt_id,
                "wav_path":    out_wav,
                "text":        text,
                "speaker":     f"spk_{dataset_name}_{safe_stem[:8]}",
                "duration":    round(actual_dur, 3),
                "source_stem": audio_stem,
                "source_dataset": dataset_name,
            })
            skipped += 1
            continue

        if actual_dur > CONFIG["max_duration_sec"]:
            segments = segment_long_audio(out_wav, output_dir,
                                           utt_id, text)
            os.remove(out_wav)
            for seg_path, seg_text in segments:
                seg_id = os.path.splitext(os.path.basename(seg_path))[0]
                seg_dur = get_duration(seg_path)
                record = {
                    "utt_id":      seg_id,
                    "wav_path":    seg_path,
                    "text":        seg_text,
                    "speaker":     f"spk_{dataset_name}_{safe_stem[:8]}",
                    "duration":    round(seg_dur, 3),
                    "source_stem": audio_stem,
                    "source_dataset": dataset_name,
                }
                if seg_dur < CONFIG["min_duration_sec"]:
                    _record_short_segment(lang, record)
                else:
                    results.append(record)
            continue

        results.append({
            "utt_id":   utt_id,
            "wav_path": out_wav,
            "text":     text,
            "speaker":  f"spk_{dataset_name}_{safe_stem[:8]}",
            "duration": round(actual_dur, 3),
            "source_stem": audio_stem,
            "source_dataset": dataset_name,
        })

    print(f"  {dataset_name}: {len(results)} segments extracted, "
          f"{skipped} skipped")
    return results


if __name__ == "__main__":
    print("Audio preprocessing script ready.")
    print("Use dataset-specific loaders below (see workflow Phase 1.5+)")
