#!/usr/bin/env python3
"""
Privacy/safety camera CLI.

Requirements:
  python -m pip install opencv-python
Optional audio support:
  python -m pip install sounddevice
  ffmpeg must be installed and available on PATH (for muxing audio into mp4)

Usage:
  python privacy_cam_cli.py
  python privacy_cam_cli.py --stop-key x --camera-index 0
"""

import argparse
import json
import os
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime

import cv2

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import msvcrt
else:
    import select
    import termios
    import tty


RUNNING = True


def handle_stop_signal(sig, frame):
    del sig, frame
    global RUNNING
    RUNNING = False


class KeyReader:
    """Reads single key presses from stdin without requiring Enter."""

    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if IS_WINDOWS:
            self.enabled = True
            return self
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        if not self.enabled:
            return None
        if IS_WINDOWS:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode("utf-8", errors="ignore")
                except Exception:
                    return None
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1)
        return None


class AudioRecorder:
    """Optional microphone recorder that writes PCM WAV in real-time."""

    def __init__(self, samplerate=44100, channels=1, device=None):
        self.path = None
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.stream = None
        self.wav = None
        self.sd = None
        self.active = False
        self.error = None
        self.lock = threading.Lock()

    def _open_wav(self, path):
        self.path = path
        self.wav = wave.open(path, "wb")
        self.wav.setnchannels(self.channels)
        self.wav.setsampwidth(2)  # int16
        self.wav.setframerate(self.samplerate)

    def start(self, path):
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = (
                "Audio requested but sounddevice is not installed. "
                "Run: python -m pip install sounddevice"
            )
            raise RuntimeError(self.error) from exc

        self.sd = sd
        self._open_wav(path)

        def callback(indata, frames, timing, status):
            del frames, timing
            if status:
                # Keep recording, but surface stream warnings to the user later.
                self.error = f"Audio stream status: {status}"
            with self.lock:
                if self.wav is not None:
                    self.wav.writeframesraw(indata)

        self.stream = self.sd.RawInputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
            device=self.device,
        )
        self.stream.start()
        self.active = True

    def rotate(self, next_path):
        with self.lock:
            old_path = self.path
            if self.wav is not None:
                self.wav.close()
            self._open_wav(next_path)
        return old_path

    def stop(self):
        closed_path = None
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.wav is not None:
            with self.lock:
                self.wav.close()
                closed_path = self.path
                self.wav = None
        self.active = False
        return closed_path


class SegmentRecorder:
    """Asynchronous segmented recorder to avoid large stop-time writes."""

    def __init__(
        self,
        output_dir,
        base_name,
        frame_size,
        fps,
        audio_enabled,
        audio_recorder,
        segment_max_bytes,
        segment_max_seconds,
        writer_queue_seconds,
    ):
        self.output_dir = output_dir
        self.base_name = base_name
        self.frame_size = frame_size
        self.fps = float(fps)
        self.audio_enabled = audio_enabled
        self.audio_recorder = audio_recorder
        self.segment_max_bytes = int(segment_max_bytes)
        self.segment_max_seconds = float(segment_max_seconds)
        queue_frames = max(30, int(self.fps * max(1.0, float(writer_queue_seconds))))
        self.frame_queue = queue.Queue(maxsize=queue_frames)
        self.finalize_queue = queue.Queue()
        self.writer_thread = None
        self.finalizer_thread = None
        self.error = None
        self.drop_count = 0
        self.drop_lock = threading.Lock()
        self.meta_lock = threading.Lock()
        self.finalized_segments = []
        self.finalizer_notes = []
        self.segment_count = 0
        self.stopping = False
        self.session_created_at = datetime.now().isoformat(timespec="seconds")
        self.manifest_path = os.path.join(self.output_dir, f"{self.base_name}_session.json")

    def _segment_paths(self, index):
        part = f"part{index:04d}"
        video_tmp = os.path.join(self.output_dir, f"{self.base_name}_{part}_video.mp4")
        audio_tmp = os.path.join(self.output_dir, f"{self.base_name}_{part}_audio.wav")
        final_mp4 = os.path.join(self.output_dir, f"{self.base_name}_{part}.mp4")
        return video_tmp, audio_tmp, final_mp4

    def start(self):
        self.finalizer_thread = threading.Thread(target=self._finalizer_loop, daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._write_manifest()
        self.finalizer_thread.start()
        self.writer_thread.start()

    def enqueue_frame(self, frame):
        if self.error is not None:
            return False
        try:
            self.frame_queue.put_nowait((frame, time.time()))
            return True
        except queue.Full:
            with self.drop_lock:
                self.drop_count += 1
            return False

    def queue_metrics(self):
        depth = self.frame_queue.qsize()
        capacity = self.frame_queue.maxsize
        percent = (depth / capacity) * 100.0 if capacity else 0.0
        return depth, capacity, percent

    def stop(self):
        if self.stopping:
            return
        self.stopping = True
        if self.writer_thread is not None and self.writer_thread.is_alive():
            while True:
                try:
                    self.frame_queue.put(None, timeout=0.2)
                    break
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
            self.writer_thread.join()
        self.finalize_queue.put(None)
        if self.finalizer_thread is not None:
            self.finalizer_thread.join()

    def _writer_loop(self):
        index = 1
        segment_frames = 0
        segment_start = time.time()
        writer = None
        video_tmp = None
        audio_tmp = None
        final_mp4 = None
        check_every = max(1, int(self.fps // 2))

        try:
            video_tmp, audio_tmp, final_mp4 = self._segment_paths(index)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_tmp, fourcc, self.fps, self.frame_size)
            if not writer.isOpened():
                raise RuntimeError("Failed to create segment writer. Try another codec/container.")
            if self.audio_enabled and self.audio_recorder is not None:
                self.audio_recorder.start(audio_tmp)

            while True:
                item = self.frame_queue.get()
                if item is None:
                    break
                frame, ts = item
                writer.write(frame)
                segment_frames += 1

                if segment_frames % check_every != 0:
                    continue

                rotate_now = False
                if self.segment_max_seconds > 0 and (ts - segment_start) >= self.segment_max_seconds:
                    rotate_now = True
                elif self.segment_max_bytes > 0:
                    try:
                        if os.path.getsize(video_tmp) >= self.segment_max_bytes:
                            rotate_now = True
                    except OSError:
                        pass

                if not rotate_now:
                    continue

                writer.release()
                used_audio = None
                next_audio_tmp = None
                if self.audio_enabled and self.audio_recorder is not None:
                    _, next_audio_tmp, _ = self._segment_paths(index + 1)
                    used_audio = self.audio_recorder.rotate(next_audio_tmp)

                self._queue_finalize(
                    index=index,
                    video_tmp=video_tmp,
                    audio_tmp=used_audio,
                    final_mp4=final_mp4,
                    frames=segment_frames,
                    duration=max(0.0, ts - segment_start),
                )
                index += 1
                segment_frames = 0
                segment_start = ts
                video_tmp, audio_tmp, final_mp4 = self._segment_paths(index)
                writer = cv2.VideoWriter(video_tmp, fourcc, self.fps, self.frame_size)
                if not writer.isOpened():
                    raise RuntimeError(
                        "Failed to create new segment writer during rotation."
                    )

            end_ts = time.time()
            if writer is not None:
                writer.release()
            if segment_frames > 0:
                used_audio = None
                if self.audio_enabled and self.audio_recorder is not None:
                    used_audio = self.audio_recorder.stop()
                self._queue_finalize(
                    index=index,
                    video_tmp=video_tmp,
                    audio_tmp=used_audio,
                    final_mp4=final_mp4,
                    frames=segment_frames,
                    duration=max(0.0, end_ts - segment_start),
                )
            else:
                if os.path.exists(video_tmp):
                    os.remove(video_tmp)
                if self.audio_enabled and self.audio_recorder is not None:
                    last_audio = self.audio_recorder.stop()
                    if last_audio and os.path.exists(last_audio):
                        os.remove(last_audio)
        except Exception as exc:
            self.error = str(exc)
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            if self.audio_enabled and self.audio_recorder is not None and self.audio_recorder.active:
                self.audio_recorder.stop()

    def _queue_finalize(self, index, video_tmp, audio_tmp, final_mp4, frames, duration):
        self.finalize_queue.put(
            {
                "index": index,
                "video_tmp": video_tmp,
                "audio_tmp": audio_tmp,
                "final_mp4": final_mp4,
                "frames": int(frames),
                "duration": float(duration),
            }
        )

    def _write_manifest(self):
        payload = {
            "session": self.base_name,
            "created_at": self.session_created_at,
            "segment_max_mb": round(self.segment_max_bytes / (1024 * 1024), 2),
            "segment_max_seconds": self.segment_max_seconds,
            "segments": self.finalized_segments,
            "notes": self.finalizer_notes,
        }
        tmp = f"{self.manifest_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def _finalizer_loop(self):
        while True:
            task = self.finalize_queue.get()
            if task is None:
                return

            try:
                note = None
                index = task["index"]
                video_tmp = task["video_tmp"]
                audio_tmp = task["audio_tmp"]
                final_mp4 = task["final_mp4"]

                if self.audio_enabled and audio_tmp:
                    mux_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_tmp,
                        "-i",
                        audio_tmp,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        final_mp4,
                    ]
                    try:
                        mux = subprocess.run(
                            mux_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        if mux.returncode == 0:
                            if os.path.exists(video_tmp):
                                os.remove(video_tmp)
                            if os.path.exists(audio_tmp):
                                os.remove(audio_tmp)
                        else:
                            os.replace(video_tmp, final_mp4)
                            note = (
                                f"segment {index}: audio mux failed, saved video-only output"
                            )
                    except FileNotFoundError:
                        os.replace(video_tmp, final_mp4)
                        note = f"segment {index}: ffmpeg not found, saved video-only output"
                else:
                    os.replace(video_tmp, final_mp4)

                segment_info = {
                    "index": index,
                    "path": final_mp4,
                    "frames": task["frames"],
                    "duration_sec": round(task["duration"], 2),
                }
                with self.meta_lock:
                    self.finalized_segments.append(segment_info)
                    self.segment_count = len(self.finalized_segments)
                    if note:
                        self.finalizer_notes.append(note)
                    self._write_manifest()
            except Exception as exc:
                with self.meta_lock:
                    self.finalizer_notes.append(
                        f"segment finalize failed: {exc}"
                    )
                    self._write_manifest()


def parse_args():
    parser = argparse.ArgumentParser(description="Privacy/safety camera CLI recorder")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=20, help="Requested frames per second")
    parser.add_argument(
        "--stop-key",
        type=str,
        default="q",
        help="Single key that stops recording and saves output",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Directory where recordings are saved",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Record microphone audio and mux with video into final mp4",
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Optional sounddevice input device name or index",
    )
    parser.add_argument(
        "--audio-samplerate",
        type=int,
        default=44100,
        help="Audio sample rate in Hz when --audio is enabled",
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=1,
        help="Number of audio channels when --audio is enabled (1=mono, 2=stereo)",
    )
    parser.add_argument(
        "--min-free-percent",
        type=float,
        default=10.0,
        help=(
            "Auto-stop when free disk space drops below this percent "
            "on the output drive (default: 10.0)"
        ),
    )
    parser.add_argument(
        "--disk-check-seconds",
        type=float,
        default=1.0,
        help="How often to check free disk space while recording (default: 1.0)",
    )
    parser.add_argument(
        "--segment-max-mb",
        type=float,
        default=950.0,
        help="Rotate to a new file when video segment reaches this size in MB (default: 950)",
    )
    parser.add_argument(
        "--segment-max-seconds",
        type=float,
        default=1200.0,
        help="Rotate to a new file after this many seconds (default: 1200)",
    )
    parser.add_argument(
        "--writer-queue-seconds",
        type=float,
        default=5.0,
        help="Approx queue depth in seconds of frames waiting to be written (default: 5.0)",
    )
    return parser.parse_args()


def main():
    global RUNNING
    RUNNING = True

    args = parse_args()
    stop_key = (args.stop_key or "q")[0]
    min_free_percent = max(0.0, min(100.0, float(args.min_free_percent)))
    disk_check_seconds = max(0.2, float(args.disk_check_seconds))
    segment_max_bytes = max(1, int(float(args.segment_max_mb) * 1024 * 1024))
    segment_max_seconds = max(1.0, float(args.segment_max_seconds))
    writer_queue_seconds = max(1.0, float(args.writer_queue_seconds))

    os.makedirs(args.output_dir, exist_ok=True)

    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera_index}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if not actual_fps or actual_fps <= 1:
        actual_fps = float(args.fps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"hotel_safety_cam_{timestamp}_{actual_w}x{actual_h}"

    audio_recorder = None
    if args.audio:
        audio_recorder = AudioRecorder(
            samplerate=args.audio_samplerate,
            channels=args.audio_channels,
            device=args.audio_device,
        )

    segment_recorder = SegmentRecorder(
        output_dir=args.output_dir,
        base_name=session_name,
        frame_size=(actual_w, actual_h),
        fps=float(actual_fps),
        audio_enabled=args.audio,
        audio_recorder=audio_recorder,
        segment_max_bytes=segment_max_bytes,
        segment_max_seconds=segment_max_seconds,
        writer_queue_seconds=writer_queue_seconds,
    )
    segment_recorder.start()

    print(f"Session: {session_name}")
    print(
        "Chunking: "
        f"{args.segment_max_mb:.0f}MB or {args.segment_max_seconds:.0f}s per segment"
    )
    print(f"Press '{stop_key}' to stop and save. Ctrl+C also stops.")

    frames = 0
    start = time.time()
    last_status = 0.0
    last_disk_check = 0.0
    stop_reason = "manual stop"

    with KeyReader() as keys:
        try:
            while RUNNING:
                ret, frame = cap.read()
                if not ret:
                    print("\nCamera frame read failed. Stopping.")
                    stop_reason = "camera frame read failed"
                    break

                segment_recorder.enqueue_frame(frame)
                frames += 1

                if segment_recorder.error:
                    RUNNING = False
                    stop_reason = f"writer error: {segment_recorder.error}"
                    break

                now = time.time()
                if now - last_disk_check >= disk_check_seconds:
                    usage = shutil.disk_usage(args.output_dir)
                    free_percent = (usage.free / usage.total) * 100 if usage.total else 0.0
                    if free_percent < min_free_percent:
                        RUNNING = False
                        stop_reason = (
                            "low disk space "
                            f"({free_percent:.2f}% free < {min_free_percent:.2f}% threshold)"
                        )
                    last_disk_check = now

                if now - last_status >= 0.5:
                    elapsed = now - start
                    marker = "REC" if int(now * 2) % 2 == 0 else "rec"
                    q_depth, q_cap, q_pct = segment_recorder.queue_metrics()
                    status = (
                        f"\r[{marker}] Recording  elapsed={elapsed:6.1f}s"
                        f"  frames={frames:<8}  audio={'on' if args.audio else 'off'}"
                        f"  segments={segment_recorder.segment_count:<4}"
                        f"  queued={q_depth}/{q_cap} ({q_pct:4.1f}%)"
                        f"  free>{min_free_percent:.1f}%"
                        f"  stop='{stop_key}' "
                    )
                    print(status, end="", flush=True)
                    last_status = now

                key = keys.read_key()
                if key and key.lower() == stop_key.lower():
                    RUNNING = False
                    stop_reason = f"stop key '{stop_key}' pressed"
        finally:
            cap.release()
            segment_recorder.stop()

    elapsed = time.time() - start
    print("\n[IDLE] Recording stopped.")
    print(f"Reason: {stop_reason}")
    print(f"Frames: {frames}")
    print(f"Elapsed: {elapsed:.1f}s")
    if args.audio and audio_recorder and audio_recorder.error:
        print(f"Audio note: {audio_recorder.error}")
    if segment_recorder.drop_count:
        print(f"Dropped frames: {segment_recorder.drop_count}")
    if segment_recorder.finalizer_notes:
        for note in segment_recorder.finalizer_notes:
            print(f"Note: {note}")
    print(f"Saved segments: {len(segment_recorder.finalized_segments)}")
    print(f"Session manifest: {segment_recorder.manifest_path}")
    if segment_recorder.finalized_segments:
        print(f"Last segment: {segment_recorder.finalized_segments[-1]['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
