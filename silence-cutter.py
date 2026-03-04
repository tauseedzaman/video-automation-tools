#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SilenceInterval:
    start: float
    end: float


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    return float(p.stdout.strip())


def print_progress_bar(iteration: float, total: float, prefix: str = 'Progress:', suffix: str = 'Complete', length: int = 40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total) if total > 0 else 0
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r', flush=True)


def detect_silence(
    input_path: str,
    silence_db: float,
    min_silence: float,
    duration: float
) -> List[SilenceInterval]:
    """
    Uses ffmpeg silencedetect to find silent segments.
    Returns list of [start, end] silence intervals.
    """
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", input_path,
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence}",
        "-f", "null", "-"
    ]
    
    print("Analyzing audio for silences...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True)
    
    silence_starts = []
    intervals: List[SilenceInterval] = []

    start_re = re.compile(r"silence_start:\s*(\d+(\.\d+)?)")
    end_re = re.compile(r"silence_end:\s*(\d+(\.\d+)?)")
    time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    stderr_output = []
    for line in process.stderr:
        stderr_output.append(line)
        
        # Check for time progress
        m_time = time_re.search(line)
        if m_time and duration > 0:
            h, m, s = float(m_time.group(1)), float(m_time.group(2)), float(m_time.group(3))
            current_time = h * 3600 + m * 60 + s
            print_progress_bar(min(current_time, duration), duration, prefix="Detecting silences", suffix="Complete")
            
        m1 = start_re.search(line)
        if m1:
            silence_starts.append(float(m1.group(1)))
            continue
        m2 = end_re.search(line)
        if m2 and silence_starts:
            end_t = float(m2.group(1))
            start_t = silence_starts.pop(0)
            if end_t > start_t:
                intervals.append(SilenceInterval(start=start_t, end=end_t))

    process.wait()
    if process.returncode != 0:
        stderr_str = "".join(stderr_output)
        raise RuntimeError(f"ffmpeg silencedetect failed:\n{stderr_str}")

    if duration > 0:
        print_progress_bar(duration, duration, prefix="Detecting silences", suffix="Complete")
    print() # Newline after progress bar

    intervals.sort(key=lambda x: x.start)
    return intervals


def merge_close(intervals: List[SilenceInterval], gap: float) -> List[SilenceInterval]:
    """Merge silence intervals that are very close."""
    if not intervals:
        return []
    merged = [intervals[0]]
    for s in intervals[1:]:
        last = merged[-1]
        if s.start <= last.end + gap:
            last.end = max(last.end, s.end)
        else:
            merged.append(s)
    return merged


def compute_keep_segments(
    duration: float,
    silences: List[SilenceInterval],
    pad: float,
    min_keep: float
) -> List[Tuple[float, float]]:
    """
    Convert silences to keep-segments (non-silent).
    pad: keep some audio around cuts by shrinking silence intervals.
    min_keep: drop tiny keep segments.
    """
    if duration <= 0:
        return []

    # Shrink silence intervals by pad (so we keep a bit around voice)
    adjusted: List[SilenceInterval] = []
    for s in silences:
        a = max(0.0, s.start + pad)
        b = min(duration, s.end - pad)
        if b > a:
            adjusted.append(SilenceInterval(a, b))

    adjusted.sort(key=lambda x: x.start)

    keep: List[Tuple[float, float]] = []
    cur = 0.0

    for s in adjusted:
        if s.start > cur:
            ks = (cur, s.start)
            if ks[1] - ks[0] >= min_keep:
                keep.append(ks)
        cur = max(cur, s.end)

    if cur < duration:
        ks = (cur, duration)
        if ks[1] - ks[0] >= min_keep:
            keep.append(ks)

    return keep


def cut_and_concat(
    input_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
    reencode: bool,
    crf: int,
    preset: str,
    enhance_audio: bool = False
) -> None:
    if not keep_segments:
        raise RuntimeError("No keep-segments found. Try increasing silence threshold or decreasing min_silence.")

    # Base audio filters for professional sound
    # afftdn: FFT Denoiser (removes background hiss/noise)
    # acompressor: Dynamic range compression (makes voice levels consistent)
    # loudnorm: EBU R128 loudness normalization (standard for YouTube/TV)
    audio_filters = ""
    if enhance_audio:
        audio_filters = "afftdn,acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=2,loudnorm"

    with tempfile.TemporaryDirectory() as tmpdir:
        parts = []
        total_parts = len(keep_segments)
        print(f"Extracting {total_parts} segments...")
        
        for i, (a, b) in enumerate(keep_segments):
            print_progress_bar(i, total_parts, prefix="Cutting", suffix=f"Segment {i+1}/{total_parts}")
            part_path = os.path.join(tmpdir, f"part_{i:04d}.mp4")
            parts.append(part_path)

            # Cutting logic
            if reencode:
                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                    "-i", input_path
                ]
                
                # Video encoding params
                cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
                
                # Audio encoding params with optional filters
                if audio_filters:
                    cmd += ["-af", audio_filters]
                
                cmd += ["-c:a", "aac", "-b:a", "192k", part_path]
            else:
                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                    "-i", input_path,
                    "-c", "copy",
                    part_path
                ]

            p = run(cmd)
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg cut failed:\n{p.stderr}")
        
        print_progress_bar(total_parts, total_parts, prefix="Cutting", suffix=f"Segment {total_parts}/{total_parts}")
        print()

        print("Concatenating segments...")
        # Concat using concat demuxer
        list_file = os.path.join(tmpdir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for part in parts:
                # concat demuxer requires this format
                f.write(f"file {shlex.quote(part)}\n")

        if reencode:
            # If already re-encoded per-part, we can copy on concat
            concat_cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]
        else:
            # Stream-copy concat should work when parts share same codec params.
            concat_cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]

        p2 = run(concat_cmd)
        if p2.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{p2.stderr}")


def main():
    ap = argparse.ArgumentParser(description="DeSilence: Automatically remove silent parts from video and enhance audio for a professional YouTube-ready output.")
    ap.add_argument("input", help="Input video file (mp4/mkv/etc)")
    ap.add_argument("-o", "--output", default=None, help="Output file (default: input_desilenced.mp4)")
    ap.add_argument("--silence_db", type=float, default=-35.0, help="Silence threshold in dB (e.g. -30, -35, -40)")
    ap.add_argument("--min_silence", type=float, default=0.45, help="Minimum silence duration to cut (seconds)")
    ap.add_argument("--pad", type=float, default=0.08, help="Keep this many seconds around speech (seconds)")
    ap.add_argument("--merge_gap", type=float, default=0.12, help="Merge silences closer than this gap (seconds)")
    ap.add_argument("--min_keep", type=float, default=0.10, help="Drop kept clips shorter than this (seconds)")
    ap.add_argument("--fast", action="store_true", help="Use stream-copy instead of re-encoding (can cause duplicate frames/clips around cuts)")
    ap.add_argument("--enhance_audio", action="store_true", help="Apply pro audio filters: noise removal (afftdn), compression, and loudness normalization")
    ap.add_argument("--crf", type=int, default=20, help="CRF when re-encoding (lower=better quality, bigger file)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset when re-encoding (ultrafast..veryslow)")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_desilenced.mp4"

    print("Fetching video duration...")
    duration = ffprobe_duration(input_path)
    print(f"Duration: {duration:.2f}s")

    silences = detect_silence(input_path, args.silence_db, args.min_silence, duration)

    # Handle possible "silence until end" case: if last silence_start exists without end
    # (silencedetect doesn't always print it). We'll ignore here; most files produce pairs.

    silences = merge_close(silences, args.merge_gap)
    keep = compute_keep_segments(duration, silences, pad=args.pad, min_keep=args.min_keep)

    print(f"Detected silence intervals: {len(silences)}")
    print(f"Keep segments: {len(keep)}")
    if len(keep) <= 25:
        for i, (a, b) in enumerate(keep):
            print(f"  keep[{i:02d}] {a:.2f} -> {b:.2f}  ({b-a:.2f}s)")

    # Force reencode if audio enhancement is requested
    reencode_final = (not args.fast) or args.enhance_audio

    cut_and_concat(
        input_path=input_path,
        keep_segments=keep,
        output_path=output_path,
        reencode=reencode_final,
        crf=args.crf,
        preset=args.preset,
        enhance_audio=args.enhance_audio
    )

    print(f"\n✅ Wrote: {output_path}")


if __name__ == "__main__":
    main()