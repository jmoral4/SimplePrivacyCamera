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
import os
import signal
import shutil
import subprocess
import sys
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

    def __init__(self, path, samplerate=44100, channels=1, device=None):
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.stream = None
        self.wav = None
        self.sd = None
        self.active = False
        self.error = None

    def start(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = (
                "Audio requested but sounddevice is not installed. "
                "Run: python -m pip install sounddevice"
            )
            raise RuntimeError(self.error) from exc

        self.sd = sd
        self.wav = wave.open(self.path, "wb")
        self.wav.setnchannels(self.channels)
        self.wav.setsampwidth(2)  # int16
        self.wav.setframerate(self.samplerate)

        def callback(indata, frames, timing, status):
            del frames, timing
            if status:
                # Keep recording, but surface stream warnings to the user later.
                self.error = f"Audio stream status: {status}"
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

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.wav is not None:
            self.wav.close()
        self.active = False


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
    return parser.parse_args()


def main():
    global RUNNING
    RUNNING = True

    args = parse_args()
    stop_key = (args.stop_key or "q")[0]
    min_free_percent = max(0.0, min(100.0, float(args.min_free_percent)))
    disk_check_seconds = max(0.2, float(args.disk_check_seconds))

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
    final_out_path = os.path.join(
        args.output_dir, f"hotel_safety_cam_{timestamp}_{actual_w}x{actual_h}.mp4"
    )
    video_path = (
        os.path.join(
            args.output_dir, f"hotel_safety_cam_{timestamp}_{actual_w}x{actual_h}_video.mp4"
        )
        if args.audio
        else final_out_path
    )
    audio_path = os.path.join(
        args.output_dir, f"hotel_safety_cam_{timestamp}_{actual_w}x{actual_h}_audio.wav"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(actual_fps), (actual_w, actual_h))
    if not writer.isOpened():
        cap.release()
        print("Failed to create output writer. Try another codec/container.")
        return 1

    audio_recorder = None
    if args.audio:
        try:
            audio_recorder = AudioRecorder(
                path=audio_path,
                samplerate=args.audio_samplerate,
                channels=args.audio_channels,
                device=args.audio_device,
            )
            audio_recorder.start()
        except Exception as exc:
            writer.release()
            cap.release()
            print(f"Failed to start audio capture: {exc}")
            return 1

    print(f"Output: {final_out_path}")
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

                writer.write(frame)
                frames += 1

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
                    status = (
                        f"\r[{marker}] Recording  elapsed={elapsed:6.1f}s"
                        f"  frames={frames:<8}  audio={'on' if args.audio else 'off'}"
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
            if audio_recorder is not None and audio_recorder.active:
                audio_recorder.stop()
            writer.release()
            cap.release()

    saved_path = final_out_path
    if args.audio:
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            final_out_path,
        ]
        try:
            mux = subprocess.run(
                mux_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if mux.returncode == 0:
                saved_path = final_out_path
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            else:
                os.replace(video_path, final_out_path)
                saved_path = final_out_path
                print("\nAudio/video mux failed. Saved video-only output.")
                if mux.stderr:
                    print("ffmpeg error summary:")
                    print(mux.stderr.strip().splitlines()[-1])
                print(f"Raw audio kept at: {audio_path}")
        except FileNotFoundError:
            os.replace(video_path, final_out_path)
            saved_path = final_out_path
            print("\nffmpeg not found. Saved video-only output.")
            print(f"Raw audio kept at: {audio_path}")

    elapsed = time.time() - start
    print("\n[IDLE] Recording stopped.")
    print(f"Reason: {stop_reason}")
    print(f"Frames: {frames}")
    print(f"Elapsed: {elapsed:.1f}s")
    if args.audio and audio_recorder and audio_recorder.error:
        print(f"Audio note: {audio_recorder.error}")
    print(f"Saved:  {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
