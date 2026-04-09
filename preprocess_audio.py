"""
preprocess_audio.py
Purpose: Convert all audio to 16kHz mono WAV, segment long files, skip short files
NEW: Undergraduate constraint — fixed seed for reproducibility (required for TA defense)
"""
import os
import sys
import subprocess
import csv
import json
from pathlib import Path
from tqdm import tqdm

# NEW: Fixed configuration for reproducibility
CONFIG = {
    "sample_rate": 16000,
    "min_duration_sec": 1.0,    # NEW: matches ESPnet2 default filter
    "max_duration_sec": 20.0,   # per bab3.tex spec
    "target_duration_sec": 10.0, # target for segmentation
    "silence_threshold_db": -50,
    "silence_min_len_ms": 500,
}

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(CONFIG["sample_rate"]),
        "-ac", "1",  # mono
        "-acodec", "pcm_s16le",
        output_path,
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

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

def segment_long_audio(input_path: str, output_dir: str,
                        utt_id: str, transcript: str) -> list:
    """
    Split audio > max_duration into segments using silence detection.
    Returns list of (segment_path, segment_transcript) tuples.
    NEW: Uses silence-based splitting per bab3.tex §Pra-pemrosesan
    """
    segments = []
    duration = get_duration(input_path)

    if duration <= CONFIG["max_duration_sec"]:
        return [(input_path, transcript)]

    # Silence-based segmentation
    segment_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"silencedetect=noise={CONFIG['silence_threshold_db']}dB:"
               f"d={CONFIG['silence_min_len_ms']/1000}",
        "-f", "null", "-"
    ]
    result = subprocess.run(segment_cmd, capture_output=True, text=True)

    # Parse silence timestamps
    silence_starts = []
    silence_ends = []
    for line in result.stderr.split('\n'):
        if 'silence_start' in line:
            try:
                t = float(line.split('silence_start: ')[1])
                silence_starts.append(t)
            except (IndexError, ValueError):
                pass
        if 'silence_end' in line:
            try:
                t = float(line.split('silence_end: ')[1].split(' ')[0])
                silence_ends.append(t)
            except (IndexError, ValueError):
                pass

    if not silence_starts:
        # No silence found — split at fixed intervals
        n_segments = int(duration / CONFIG["target_duration_sec"]) + 1
        for i in range(n_segments):
            start = i * CONFIG["target_duration_sec"]
            seg_path = os.path.join(output_dir, f"{utt_id}_seg{i:03d}.wav")
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-ss", str(start),
                "-t", str(CONFIG["target_duration_sec"]),
                "-ar", str(CONFIG["sample_rate"]),
                "-ac", "1", "-acodec", "pcm_s16le",
                seg_path, "-loglevel", "error"
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(seg_path):
                seg_dur = get_duration(seg_path)
                if seg_dur >= CONFIG["min_duration_sec"]:
                    segments.append((seg_path, transcript))
        return segments

    # Build split points from silence midpoints
    split_points = [0.0]
    for s, e in zip(silence_starts, silence_ends):
        mid = (s + e) / 2.0
        split_points.append(mid)
    split_points.append(duration)

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        seg_dur = end - start
        if seg_dur < CONFIG["min_duration_sec"]:
            continue
        if seg_dur > CONFIG["max_duration_sec"]:
            # Subdivide further at fixed intervals
            for j in range(int(seg_dur / CONFIG["target_duration_sec"]) + 1):
                sub_start = start + j * CONFIG["target_duration_sec"]
                sub_dur = min(CONFIG["target_duration_sec"], end - sub_start)
                if sub_dur < CONFIG["min_duration_sec"]:
                    continue
                seg_path = os.path.join(output_dir,
                                         f"{utt_id}_seg{i:03d}s{j:03d}.wav")
                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-ss", str(sub_start), "-t", str(sub_dur),
                    "-ar", str(CONFIG["sample_rate"]), "-ac", "1",
                    "-acodec", "pcm_s16le", seg_path, "-loglevel", "error"
                ]
                subprocess.run(cmd, capture_output=True)
                if os.path.exists(seg_path):
                    segments.append((seg_path, transcript))
            continue

        seg_path = os.path.join(output_dir, f"{utt_id}_seg{i:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start), "-t", str(seg_dur),
            "-ar", str(CONFIG["sample_rate"]), "-ac", "1",
            "-acodec", "pcm_s16le", seg_path, "-loglevel", "error"
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(seg_path):
            segments.append((seg_path, transcript))

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
            os.remove(out_wav)
            skipped += 1
            continue

        if duration > CONFIG["max_duration_sec"]:
            segments = segment_long_audio(out_wav, output_dir,
                                           utt_id, transcript)
            os.remove(out_wav)  # remove the unsegmented version
            for seg_path, seg_text in segments:
                seg_id = os.path.splitext(os.path.basename(seg_path))[0]
                seg_dur = get_duration(seg_path)
                records.append({
                    "utt_id": seg_id,
                    "wav_path": seg_path,
                    "text": seg_text.strip().lower() if lang != "ar" else seg_text.strip(),
                    "speaker": f"spk_{dataset_name}_{stem[:6]}",
                    "duration": round(seg_dur, 3)
                })
        else:
            records.append({
                "utt_id": utt_id,
                "wav_path": out_wav,
                "text": transcript.strip().lower() if lang != "ar" else transcript.strip(),
                "speaker": f"spk_{dataset_name}_{stem[:6]}",
                "duration": round(duration, 3)
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

        # Filter by duration
        if duration < CONFIG["min_duration_sec"]:
            skipped += 1
            continue
        if duration > CONFIG["max_duration_sec"]:
            # Segment too long — skip; podcast segments should be short
            # NEW: max_duration=20s enforced; long segments discarded not split
            #      to preserve transcript integrity (no mid-sentence split)
            skipped += 1
            continue

        # Match audio file
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

        # Sanitize utterance ID
        safe_stem = audio_stem.replace(' ', '_').replace('/', '_')
        utt_id    = f"{lang}_{dataset_name}_{safe_stem}_s{i:05d}"
        out_wav   = os.path.join(output_dir, f"{utt_id}.wav")

        # Extract segment with ffmpeg
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
            os.remove(out_wav)
            skipped += 1
            continue

        results.append({
            "utt_id":   utt_id,
            "wav_path": out_wav,
            "text":     text,
            "speaker":  f"spk_{dataset_name}_{safe_stem[:8]}",
            "duration": round(actual_dur, 3)
        })

    print(f"  {dataset_name}: {len(results)} segments extracted, "
          f"{skipped} skipped")
    return results


if __name__ == "__main__":
    print("Audio preprocessing script ready.")
    print("Use dataset-specific loaders below (see workflow Phase 1.5+)")
