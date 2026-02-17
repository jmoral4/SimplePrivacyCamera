#!/usr/bin/env python3
"""Analyze MP4 recordings for notable events and build a clickable timeline report."""

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from statistics import fmean, median, pstdev
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


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
    parser.add_argument("video", nargs="?", help="Path to MP4 video")
    parser.add_argument(
        "--output-dir",
        default="scrub_reports",
        help="Directory to write JSON + HTML outputs (default: scrub_reports)",
    )
    parser.add_argument(
        "--regen-from-json",
        default=None,
        help="Regenerate HTML only from an existing *_events_*.json file",
    )
    parser.add_argument(
        "--regen-from-output-dir",
        default=None,
        help="Regenerate HTML only from newest *_events_*.json in this directory",
    )
    parser.add_argument(
        "--html-video-src",
        default=None,
        help="Optional video path/URL to embed in regenerated HTML",
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
    parser.add_argument(
        "--person-model",
        default="models/yolo11n.onnx",
        help="Path to ONNX person-capable detector (default: models/yolo11n.onnx)",
    )
    parser.add_argument(
        "--person-conf-threshold",
        type=float,
        default=0.20,
        help="Confidence threshold for person detection (default: 0.20)",
    )
    parser.add_argument(
        "--person-nms-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold for person detection (default: 0.45)",
    )
    parser.add_argument(
        "--person-input-size",
        type=int,
        default=640,
        help="Model input size for ONNX detector (default: 640)",
    )
    parser.add_argument(
        "--person-bottom-roi",
        type=float,
        default=0.45,
        help="Fraction of frame height for extra bottom ROI pass (default: 0.45)",
    )
    parser.add_argument(
        "--person-roi-upscale",
        type=float,
        default=1.6,
        help="Upscale factor for bottom ROI before detection (default: 1.6)",
    )
    parser.add_argument(
        "--require-people-model",
        action="store_true",
        help="Fail if ONNX person model cannot be loaded instead of falling back to HOG",
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
    person_model: str,
    person_conf_threshold: float,
    person_nms_threshold: float,
    person_input_size: int,
    person_bottom_roi: float,
    person_roi_upscale: float,
    require_people_model: bool,
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

    def clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def nms_xywh(
        boxes: Sequence[Tuple[int, int, int, int]],
        scores: Sequence[float],
        conf_threshold: float,
        nms_threshold: float,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        if not boxes:
            return [], []
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[int(x), int(y), int(w), int(h)] for (x, y, w, h) in boxes],
            scores=[float(s) for s in scores],
            score_threshold=float(conf_threshold),
            nms_threshold=float(nms_threshold),
        )
        if idxs is None or len(idxs) == 0:
            return [], []
        flat: List[int] = []
        for idx in idxs:
            if isinstance(idx, (list, tuple)):
                flat.append(int(idx[0]))
            else:
                flat.append(int(idx))
        out_boxes = [boxes[i] for i in flat]
        out_scores = [float(scores[i]) for i in flat]
        return out_boxes, out_scores

    def decode_yolo_person_predictions(
        raw_preds,
        image_w: int,
        image_h: int,
        conf_threshold: float,
        input_size: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        preds = raw_preds
        if isinstance(preds, (list, tuple)):
            if not preds:
                return [], []
            preds = preds[0]
        if preds is None or not hasattr(preds, "shape"):
            return [], []
        if len(preds.shape) == 3:
            preds = preds[0]
        if len(preds.shape) != 2:
            return [], []

        # Ultralytics ONNX frequently exports as [84, N]; transpose to [N, 84].
        if preds.shape[0] < preds.shape[1]:
            preds = preds.transpose()
        if preds.shape[1] < 5:
            return [], []

        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        for row in preds:
            # Supports YOLOv8/v11 style [cx, cy, w, h, cls...].
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4:]
            if class_scores.size == 0:
                continue

            person_score = float(class_scores[0])  # COCO class 0 = person
            if person_score < conf_threshold:
                continue

            # Handle either normalized [0,1] or input-scale [0,input_size] coords.
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.5:
                x = int((cx - bw / 2.0) * image_w)
                y = int((cy - bh / 2.0) * image_h)
                ww = int(bw * image_w)
                hh = int(bh * image_h)
            else:
                sx = float(image_w) / float(input_size)
                sy = float(image_h) / float(input_size)
                x = int((cx - bw / 2.0) * sx)
                y = int((cy - bh / 2.0) * sy)
                ww = int(bw * sx)
                hh = int(bh * sy)

            if ww <= 2 or hh <= 2:
                continue

            x = int(clamp(float(x), 0.0, float(max(0, image_w - 1))))
            y = int(clamp(float(y), 0.0, float(max(0, image_h - 1))))
            ww = int(clamp(float(ww), 1.0, float(max(1, image_w - x))))
            hh = int(clamp(float(hh), 1.0, float(max(1, image_h - y))))

            boxes.append((x, y, ww, hh))
            scores.append(person_score)

        return boxes, scores

    def run_yolo_person_pass_cv2(
        net,
        image_bgr,
        conf_threshold: float,
        input_size: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return [], []

        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(int(input_size), int(input_size)),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        outs = net.forward()
        return decode_yolo_person_predictions(
            raw_preds=outs,
            image_w=w,
            image_h=h,
            conf_threshold=conf_threshold,
            input_size=input_size,
        )

    def run_yolo_person_pass_ort(
        session,
        input_name: str,
        image_bgr,
        conf_threshold: float,
        input_size: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return [], []

        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(int(input_size), int(input_size)),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        outputs = session.run(None, {input_name: blob})
        return decode_yolo_person_predictions(
            raw_preds=outputs,
            image_w=w,
            image_h=h,
            conf_threshold=conf_threshold,
            input_size=input_size,
        )

    def detect_people_with_runner(
        pass_runner: Callable[[object, float, int], Tuple[List[Tuple[int, int, int, int]], List[float]]],
        frame_bgr,
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
        bottom_roi: float,
        roi_upscale: float,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        h, _ = frame_bgr.shape[:2]
        all_boxes, all_scores = pass_runner(frame_bgr, conf_threshold, input_size)

        bottom_roi = clamp(bottom_roi, 0.0, 1.0)
        roi_upscale = max(1.0, float(roi_upscale))
        if bottom_roi > 0.0:
            roi_h = int(max(1, round(h * bottom_roi)))
            y0 = max(0, h - roi_h)
            roi = frame_bgr[y0:h, :]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_input = roi
                if roi_upscale > 1.0:
                    roi_input = cv2.resize(
                        roi,
                        None,
                        fx=roi_upscale,
                        fy=roi_upscale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                roi_boxes, roi_scores = pass_runner(roi_input, conf_threshold, input_size)
                for (x, y, ww, hh), score in zip(roi_boxes, roi_scores):
                    if roi_upscale > 1.0:
                        x = int(round(x / roi_upscale))
                        y = int(round(y / roi_upscale))
                        ww = int(round(ww / roi_upscale))
                        hh = int(round(hh / roi_upscale))
                    all_boxes.append((x, y + y0, ww, hh))
                    all_scores.append(score)

        return nms_xywh(
            boxes=all_boxes,
            scores=all_scores,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

    hog = None
    dnn_net = None
    ort_session = None
    ort_input_name: Optional[str] = None
    person_detector = "disabled"
    if not skip_people:
        model_path = os.path.abspath(person_model)
        if os.path.exists(model_path):
            load_errors: List[str] = []
            try:
                dnn_net = cv2.dnn.readNet(model_path)
                person_detector = "dnn"
                print(f"Person detector: DNN (model={model_path})")
            except Exception as exc:
                dnn_net = None
                load_errors.append(f"OpenCV DNN load failed: {exc}")

            if dnn_net is None and ort is not None:
                try:
                    ort_session = ort.InferenceSession(
                        model_path,
                        providers=["CPUExecutionProvider"],
                    )
                    inputs = ort_session.get_inputs()
                    if not inputs:
                        raise RuntimeError("ONNX model has no inputs")
                    ort_input_name = inputs[0].name
                    person_detector = "onnxruntime"
                    print(f"Person detector: ONNX Runtime (model={model_path})")
                except Exception as exc:
                    ort_session = None
                    ort_input_name = None
                    load_errors.append(f"ONNX Runtime load failed: {exc}")
            elif dnn_net is None:
                load_errors.append(
                    "ONNX Runtime not installed. Install with: python -m pip install onnxruntime"
                )

            if dnn_net is None and ort_session is None:
                print(f"Person detector: failed to load ONNX model at {model_path}")
                for err in load_errors:
                    print(f"  - {err}")
                if require_people_model:
                    raise RuntimeError("Failed to initialize person ONNX model")
                print("Person detector: falling back to HOG")
        else:
            msg = f"Person detector: model file not found at {model_path}"
            if require_people_model:
                raise RuntimeError(msg)
            print(f"{msg}; falling back to HOG")

        if dnn_net is None and ort_session is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            person_detector = "hog"
            print("Person detector: HOG fallback active")
    else:
        print("Person detector: disabled (--skip-people)")

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

        if not skip_people and frame_idx % people_every_frames == 0:
            person_count = 0
            confidence = 0.0
            if dnn_net is not None or (ort_session is not None and ort_input_name is not None):
                try:
                    if dnn_net is not None:
                        pass_runner = lambda image, conf, inp: run_yolo_person_pass_cv2(
                            dnn_net, image, conf, inp
                        )
                    else:
                        pass_runner = lambda image, conf, inp: run_yolo_person_pass_ort(
                            ort_session, ort_input_name, image, conf, inp
                        )

                    person_boxes, person_scores = detect_people_with_runner(
                        pass_runner=pass_runner,
                        frame_bgr=frame,
                        conf_threshold=max(0.01, float(person_conf_threshold)),
                        nms_threshold=clamp(float(person_nms_threshold), 0.05, 0.95),
                        input_size=max(160, int(person_input_size)),
                        bottom_roi=float(person_bottom_roi),
                        roi_upscale=max(1.0, float(person_roi_upscale)),
                    )
                    person_count = len(person_boxes)
                    confidence = max(person_scores, default=0.0)
                except Exception as exc:
                    print(
                        f"Warning: {person_detector} person detection failed at {current_time:.2f}s "
                        f"({exc}). Falling back to HOG."
                    )
                    dnn_net = None
                    ort_session = None
                    ort_input_name = None
                    if hog is None:
                        hog = cv2.HOGDescriptor()
                        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                    person_detector = "hog"
            elif hog is not None:
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
                confidence = max([float(w) for w in weights], default=0.0)

            if person_count > 0 and current_time - last_event_by_type["person"] >= cooldown_sec:
                details = (
                    f"Detected {person_count} person(s)"
                    f" [{person_detector}, conf={confidence:.2f}]"
                )
                events.append(
                    Event(
                        time_sec=current_time,
                        event_type="person",
                        score=float(confidence),
                        details=details,
                    )
                )
                last_event_by_type["person"] = current_time

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

        prev_gray = gray
        prev_hist = hist

    cap.release()

    events.sort(key=lambda e: e.time_sec)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "events": events,
        "person_detector": person_detector,
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
    people_marker_divs = []
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
        if event.event_type == "person":
            people_marker_divs.append(
                f'<div class="marker marker-person" style="left:{pct:.3f}%;" '
                f'data-people-time="{event.time_sec:.3f}" '
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
.people-wrap {{ margin: 6px 0 10px; }}
.people-title {{ color: var(--muted); font-size: 0.85rem; margin: 6px 0 6px; }}
.timeline.people {{ height: 28px; background: #15284f; border-color: #4167a3; }}
.marker-person {{ top: 2px; width: 8px; height: 22px; border-radius: 4px; background: #2563eb; }}
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
        <button onclick="jumpToPerson(-1)">Prev Person</button>
        <button onclick="jumpToPerson(1)">Next Person</button>
      </div>
      <div class="timeline" id="timeline">{''.join(marker_divs)}</div>
      <div class="people-wrap">
        <div class="people-title">People Scrub Timeline</div>
        <div class="timeline people" id="peopleTimeline">{''.join(people_marker_divs)}</div>
      </div>
      <div class="legend">Click a marker or event row to seek playback.</div>
      <div class="list">{''.join(event_rows) if event_rows else '<div class="meta">No events detected.</div>'}</div>
    </div>
  </div>
<script>
const player = document.getElementById('player');
const peopleTimes = Array.from(document.querySelectorAll('[data-people-time]'))
  .map(el => Number(el.getAttribute('data-people-time')))
  .filter(Number.isFinite)
  .sort((a, b) => a - b);
function seekTo(seconds) {{
  player.currentTime = seconds;
  player.play().catch(() => {{}});
}}
function jump(delta) {{
  const next = Math.max(0, player.currentTime + delta);
  player.currentTime = next;
}}
function jumpToPerson(direction) {{
  if (!peopleTimes.length) return;
  const now = player.currentTime || 0;
  if (direction > 0) {{
    const next = peopleTimes.find(t => t > now + 0.05);
    seekTo(next ?? peopleTimes[0]);
    return;
  }}
  for (let i = peopleTimes.length - 1; i >= 0; i -= 1) {{
    if (peopleTimes[i] < now - 0.05) {{
      seekTo(peopleTimes[i]);
      return;
    }}
  }}
  seekTo(peopleTimes[peopleTimes.length - 1]);
}}
</script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_events_filename(json_path: str) -> Tuple[str, str]:
    name = os.path.basename(json_path)
    match = re.match(r"(.+)_events_(\d{8}_\d{6})\.json$", name)
    if not match:
        raise RuntimeError(
            f"Expected file named like *_events_YYYYMMDD_HHMMSS.json, got: {name}"
        )
    return match.group(1), match.group(2)


def find_latest_events_json(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    candidates: List[str] = []
    for name in os.listdir(output_dir):
        if re.match(r".+_events_\d{8}_\d{6}\.json$", name):
            candidates.append(os.path.join(output_dir, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def event_from_dict(raw: Dict[str, object]) -> Event:
    return Event(
        time_sec=float(raw.get("time_sec", 0.0)),
        event_type=str(raw.get("event_type", "unknown")),
        score=float(raw.get("score", 0.0)),
        details=str(raw.get("details", "")),
    )


def regen_html_only(
    output_dir: str,
    regen_json: Optional[str],
    regen_output_dir: Optional[str],
    html_video_src: Optional[str],
) -> int:
    if regen_json:
        json_path = os.path.abspath(regen_json)
    else:
        search_dir = os.path.abspath(regen_output_dir or output_dir)
        latest = find_latest_events_json(search_dir)
        if not latest:
            print(f"No event JSON found in: {search_dir}")
            return 1
        json_path = latest

    if not os.path.exists(json_path):
        print(f"Event JSON not found: {json_path}")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        print(f"Invalid events JSON format: {json_path}")
        return 1

    events = [event_from_dict(e) for e in raw_events if isinstance(e, dict)]
    events.sort(key=lambda e: e.time_sec)

    base, stamp = parse_events_filename(json_path)
    html_path = os.path.join(os.path.dirname(json_path), f"{base}_timeline_{stamp}.html")

    report_video_path = ""
    if html_video_src:
        report_video_path = os.path.abspath(html_video_src)
    else:
        candidate_browser = os.path.join(
            os.path.dirname(json_path), f"{base}_browser_{stamp}.mp4"
        )
        source_video = str(payload.get("source_video", ""))
        if os.path.exists(candidate_browser):
            report_video_path = candidate_browser
        elif source_video and os.path.exists(source_video):
            report_video_path = source_video

    if report_video_path:
        video_src = os.path.relpath(report_video_path, os.path.dirname(html_path))
    else:
        video_src = ""

    source_video_name = os.path.basename(str(payload.get("source_video", ""))) or "unknown"
    duration_sec = float(payload.get("duration_sec", 0.0))
    if duration_sec <= 0.0 and events:
        duration_sec = max(e.time_sec for e in events)

    generate_html_report(
        report_path=html_path,
        video_src=video_src,
        duration_sec=duration_sec,
        events=events,
        source_video_name=source_video_name,
    )
    print(f"Regenerated timeline report: {html_path}")
    print(f"Source events JSON: {json_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.regen_from_json or args.regen_from_output_dir:
        return regen_html_only(
            output_dir=args.output_dir,
            regen_json=args.regen_from_json,
            regen_output_dir=args.regen_from_output_dir,
            html_video_src=args.html_video_src,
        )

    if not args.video:
        print("Missing input video path. Provide VIDEO or use --regen-from-json/--regen-from-output-dir.")
        return 1

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
        person_model=args.person_model,
        person_conf_threshold=args.person_conf_threshold,
        person_nms_threshold=args.person_nms_threshold,
        person_input_size=max(160, args.person_input_size),
        person_bottom_roi=max(0.0, min(1.0, args.person_bottom_roi)),
        person_roi_upscale=max(1.0, args.person_roi_upscale),
        require_people_model=args.require_people_model,
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
            "person_detector": visual.get("person_detector", "unknown"),
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
