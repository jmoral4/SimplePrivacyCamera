#!/usr/bin/env python3
"""Analyze MP4 recordings for notable events and build a clickable timeline report."""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from statistics import fmean, median, pstdev
from typing import Dict, List, Optional

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


@dataclass
class Event:
    time_sec: float
    event_type: str
    score: float
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-scrub MP4 video for visual/audio events and generate a timeline report"
    )
    parser.add_argument("video", help="Path to MP4 video")
    parser.add_argument(
        "--output-dir",
        default="scrub_reports",
        help="Directory to write JSON + HTML outputs (default: scrub_reports)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frame sample rate for visual analysis (default: 2.0)",
    )
    parser.add_argument(
        "--people-every-sec",
        type=float,
        default=1.0,
        help="How often to run person detection in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=12.0,
        help="Threshold for motion spike score (default: 12.0)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.22,
        help="Threshold for scene-change score using histogram delta (default: 0.22)",
    )
    parser.add_argument(
        "--audio-window-ms",
        type=int,
        default=250,
        help="Audio RMS window size in milliseconds (default: 250)",
    )
    parser.add_argument(
        "--audio-zscore",
        type=float,
        default=2.5,
        help="Z-score threshold for audio spikes (default: 2.5)",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=1.0,
        help="Minimum seconds between events of same type (default: 1.0)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional cap on analyzed duration for quick runs",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio analysis even if ffmpeg exists",
    )
    parser.add_argument(
        "--skip-people",
        action="store_true",
        help="Skip person detection step",
    )
    return parser.parse_args()


def ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000.0))
    return f"{mins:02d}:{secs:02d}.{ms:03d}"


def extract_audio_to_wav(video_path: str, wav_path: str, sample_rate: int = 16000) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(wav_path)


def make_browser_compatible_video(video_path: str, out_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


def analyze_audio_spikes(
    wav_path: str,
    window_ms: int,
    z_threshold: float,
    cooldown_sec: float,
) -> List[Event]:
    events: List[Event] = []
    with wave.open(wav_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        total_frames = wav_file.getnframes()

        if sample_width != 2:
            return events

        window_frames = max(1, int(framerate * (window_ms / 1000.0)))
        rms_values: List[float] = []
        times: List[float] = []

        frame_index = 0
        while frame_index < total_frames:
            raw = wav_file.readframes(window_frames)
            if not raw:
                break
            sample_count = len(raw) // sample_width
            if sample_count == 0:
                break

            sum_sq = 0.0
            included = 0
            for sample_idx, (val,) in enumerate(struct.iter_unpack("<h", raw)):
                if channels > 1:
                    # Audio was requested mono, but keep code robust.
                    if sample_idx % channels != 0:
                        continue
                sum_sq += float(val) * float(val)
                included += 1

            if included == 0:
                break

            rms = math.sqrt(sum_sq / float(included))
            midpoint = frame_index + min(window_frames, total_frames - frame_index) / 2.0
            time_sec = midpoint / float(framerate)

            rms_values.append(rms)
            times.append(time_sec)
            frame_index += window_frames

    if len(rms_values) < 4:
        return events

    mu = fmean(rms_values)
    sigma = pstdev(rms_values) or 1.0
    baseline = median(rms_values)
    dynamic_floor = max(mu + z_threshold * sigma, baseline * 2.0)

    last_time = -9999.0
    for idx, rms in enumerate(rms_values):
        if rms < dynamic_floor:
            continue
        when = times[idx]
        if when - last_time < cooldown_sec:
            continue
        z_score = (rms - mu) / sigma
        events.append(
            Event(
                time_sec=when,
                event_type="audio_spike",
                score=float(z_score),
                details=f"Audio RMS spike (rms={rms:.1f}, z={z_score:.2f})",
            )
        )
        last_time = when

    return events


def analyze_video(
    video_path: str,
    sample_fps: float,
    people_every_sec: float,
    motion_threshold: float,
    scene_threshold: float,
    cooldown_sec: float,
    max_seconds: Optional[float],
    skip_people: bool,
) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0.1:
        fps = 20.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    if max_seconds is not None:
        duration_sec = min(duration_sec, max_seconds)

    sample_every_frames = max(1, int(round(fps / max(sample_fps, 0.1))))
    people_every_frames = max(1, int(round(fps * max(people_every_sec, 0.1))))

    hog = None
    if not skip_people:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    events: List[Event] = []
    prev_gray = None
    prev_hist = None
    frame_idx = 0
    last_event_by_type = {
        "motion": -9999.0,
        "scene_change": -9999.0,
        "person": -9999.0,
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        current_time = frame_idx / fps
        if max_seconds is not None and current_time > max_seconds:
            break

        if frame_idx % sample_every_frames != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            delta = cv2.absdiff(gray, prev_gray)
            motion_score = float(cv2.mean(delta)[0])
            if motion_score >= motion_threshold and current_time - last_event_by_type["motion"] >= cooldown_sec:
                events.append(
                    Event(
                        time_sec=current_time,
                        event_type="motion",
                        score=motion_score,
                        details=f"Motion spike (score={motion_score:.2f})",
                    )
                )
                last_event_by_type["motion"] = current_time

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
            scene_score = max(0.0, 1.0 - corr)
            if scene_score >= scene_threshold and current_time - last_event_by_type["scene_change"] >= cooldown_sec:
                events.append(
                    Event(
                        time_sec=current_time,
                        event_type="scene_change",
                        score=scene_score,
                        details=f"Scene change (delta={scene_score:.3f})",
                    )
                )
                last_event_by_type["scene_change"] = current_time

        if hog is not None and frame_idx % people_every_frames == 0:
            display = frame
            if frame.shape[1] > 640:
                scale = 640.0 / float(frame.shape[1])
                display = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

            rects, weights = hog.detectMultiScale(
                display,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            person_count = int(len(rects))
            if person_count > 0 and current_time - last_event_by_type["person"] >= cooldown_sec:
                confidence = max([float(w) for w in weights], default=0.0)
                events.append(
                    Event(
                        time_sec=current_time,
                        event_type="person",
                        score=confidence,
                        details=f"Detected {person_count} person(s)",
                    )
                )
                last_event_by_type["person"] = current_time

        prev_gray = gray
        prev_hist = hist

    cap.release()

    events.sort(key=lambda e: e.time_sec)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "events": events,
    }


def generate_html_report(
    report_path: str,
    video_src: str,
    duration_sec: float,
    events: List[Event],
    source_video_name: str,
) -> None:
    colors = {
        "motion": "#ef4444",
        "scene_change": "#f59e0b",
        "person": "#2563eb",
        "audio_spike": "#10b981",
    }

    event_rows = []
    marker_divs = []
    for event in events:
        pct = (event.time_sec / duration_sec * 100.0) if duration_sec > 0 else 0.0
        pct = max(0.0, min(100.0, pct))
        color = colors.get(event.event_type, "#64748b")
        label = f"[{event.event_type}] {ts(event.time_sec)} - {event.details}"
        event_rows.append(
            f'<button class="event-row" onclick="seekTo({event.time_sec:.3f})"'
            f' title="score={event.score:.3f}"><span class="pill" style="background:{color}"></span>'
            f"{escape(label)}</button>"
        )
        marker_divs.append(
            f'<div class="marker" style="left:{pct:.3f}%; background:{color};" '
            f'onclick="seekTo({event.time_sec:.3f})" title="{escape(label)}"></div>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Video Event Timeline</title>
<style>
:root {{
  --bg: #0b1020;
  --panel: #121a30;
  --ink: #e5e7eb;
  --muted: #94a3b8;
  --track: #1f2a44;
  --accent: #38bdf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: radial-gradient(circle at top, #18274f 0%, var(--bg) 45%);
  color: var(--ink);
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 20px; }}
.panel {{ background: color-mix(in oklab, var(--panel) 92%, black); border: 1px solid #2b3a63; border-radius: 12px; padding: 14px; }}
h1 {{ margin: 0 0 8px; font-size: 1.2rem; }}
.meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 12px; }}
video {{ width: 100%; border-radius: 10px; border: 1px solid #3b4e7e; background: black; }}
.timeline {{ position: relative; height: 24px; border-radius: 999px; margin: 12px 0 8px; background: var(--track); border: 1px solid #32456f; }}
.marker {{ position: absolute; top: 2px; width: 6px; height: 18px; border-radius: 3px; cursor: pointer; transform: translateX(-50%); }}
.legend {{ color: var(--muted); font-size: 0.85rem; margin: 8px 0 10px; }}
.list {{ display: grid; grid-template-columns: 1fr; gap: 8px; max-height: 300px; overflow-y: auto; }}
.event-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  text-align: left;
  border: 1px solid #2f416a;
  background: #111c37;
  color: var(--ink);
  border-radius: 10px;
  padding: 9px 10px;
  cursor: pointer;
}}
.event-row:hover {{ border-color: var(--accent); }}
.pill {{ width: 10px; height: 10px; border-radius: 99px; display: inline-block; flex-shrink: 0; }}
.controls {{ display: flex; gap: 8px; margin: 10px 0; flex-wrap: wrap; }}
.controls button {{
  border: 1px solid #355a8b;
  background: #0f244b;
  color: var(--ink);
  border-radius: 8px;
  padding: 6px 10px;
  cursor: pointer;
}}
</style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Event Timeline</h1>
      <div class="meta">Source: {escape(source_video_name)} | Duration: {ts(duration_sec)} | Events: {len(events)}</div>
      <video id="player" controls preload="metadata" src="{escape(video_src)}"></video>
      <div class="controls">
        <button onclick="jump(-5)">-5s</button>
        <button onclick="jump(-1)">-1s</button>
        <button onclick="jump(1)">+1s</button>
        <button onclick="jump(5)">+5s</button>
      </div>
      <div class="timeline" id="timeline">{''.join(marker_divs)}</div>
      <div class="legend">Click a marker or event row to seek playback.</div>
      <div class="list">{''.join(event_rows) if event_rows else '<div class="meta">No events detected.</div>'}</div>
    </div>
  </div>
<script>
const player = document.getElementById('player');
function seekTo(seconds) {{
  player.currentTime = seconds;
  player.play().catch(() => {{}});
}}
function jump(delta) {{
  const next = Math.max(0, player.currentTime + delta);
  player.currentTime = next;
}}
</script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    args = parse_args()
    if cv2 is None:
        print("Missing dependency: opencv-python")
        print("Install with: python3 -m pip install opencv-python")
        return 1

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    visual = analyze_video(
        video_path=video_path,
        sample_fps=max(0.1, args.sample_fps),
        people_every_sec=max(0.1, args.people_every_sec),
        motion_threshold=args.motion_threshold,
        scene_threshold=args.scene_threshold,
        cooldown_sec=max(0.1, args.cooldown_sec),
        max_seconds=args.max_seconds,
        skip_people=args.skip_people,
    )
    events: List[Event] = list(visual["events"])

    audio_notes = "audio not requested"
    if not args.skip_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            if extract_audio_to_wav(video_path, wav_path):
                audio_events = analyze_audio_spikes(
                    wav_path=wav_path,
                    window_ms=max(50, args.audio_window_ms),
                    z_threshold=args.audio_zscore,
                    cooldown_sec=max(0.1, args.cooldown_sec),
                )
                events.extend(audio_events)
                audio_notes = f"audio analyzed ({len(audio_events)} spikes)"
            else:
                audio_notes = "ffmpeg missing or no audio track; audio analysis skipped"
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    events.sort(key=lambda e: e.time_sec)

    base = os.path.splitext(os.path.basename(video_path))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"{base}_events_{stamp}.json")
    html_path = os.path.join(args.output_dir, f"{base}_timeline_{stamp}.html")
    browser_video_path = os.path.join(args.output_dir, f"{base}_browser_{stamp}.mp4")

    payload = {
        "source_video": video_path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": visual["duration_sec"],
        "fps": visual["fps"],
        "frame_count": visual["frame_count"],
        "summary": {
            "event_count": len(events),
            "motion": sum(1 for e in events if e.event_type == "motion"),
            "scene_change": sum(1 for e in events if e.event_type == "scene_change"),
            "person": sum(1 for e in events if e.event_type == "person"),
            "audio_spike": sum(1 for e in events if e.event_type == "audio_spike"),
            "audio_notes": audio_notes,
        },
        "events": [asdict(e) for e in events],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    report_video_path = video_path
    browser_video_notes = "using original source video"
    if make_browser_compatible_video(video_path, browser_video_path):
        report_video_path = browser_video_path
        browser_video_notes = f"generated browser-compatible video: {browser_video_path}"
    elif shutil.which("ffmpeg"):
        browser_video_notes = "ffmpeg available, but browser-compatible transcode failed; using original source"
    else:
        browser_video_notes = "ffmpeg not available; using original source video"

    video_src = os.path.relpath(report_video_path, os.path.dirname(html_path))
    generate_html_report(
        report_path=html_path,
        video_src=video_src,
        duration_sec=float(visual["duration_sec"]),
        events=events,
        source_video_name=os.path.basename(video_path),
    )

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote timeline report: {html_path}")
    print(browser_video_notes)
    print(f"Total events: {len(events)} ({audio_notes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
