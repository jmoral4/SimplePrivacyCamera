# Hotel Safety Cam (CLI)

A small privacy/safety camera recorder for hotel rooms or temporary spaces.

It records from your webcam, shows a clear terminal recording state, and lets you stop recording with a single key while safely writing the video to disk.

## Features

- Clear status while active: blinking `[REC]/[rec]` in terminal.
- Clear status when stopped: `[IDLE]`.
- Keyboard stop key (default `q`) without pressing Enter.
- Also supports `Ctrl+C` to stop and save.
- Writes timestamped `.mp4` files to disk.

## Requirements

- Python 3.8+
- Webcam/camera device
- `opencv-python`
- Optional for audio capture: `sounddevice`
- Optional for combining audio+video into one `.mp4`: `ffmpeg` on PATH

Install dependency:

```bash
python3 -m pip install opencv-python
```

Optional audio dependency:

```bash
python3 -m pip install sounddevice
```

For event scrubbing reports, `ffmpeg` is optional but recommended (used for audio spike detection).

## Run

```bash
python3 privacy_cam_cli.py
```

Default behavior:

- Uses camera index `0`
- Requests `1280x720` at `20` FPS
- Stop key is `q`
- Saves to `recordings/`

## CLI Flags

```text
--camera-index INT   Camera device index (default: 0)
--width INT          Requested frame width (default: 1280)
--height INT         Requested frame height (default: 720)
--fps INT            Requested frames per second (default: 20)
--stop-key STR       Single key to stop recording and save (default: q)
--output-dir STR     Output directory for saved videos (default: recordings)
--audio              Enable microphone recording and mux into mp4
--audio-device STR   Optional input device name/index for sounddevice
--audio-samplerate   Audio sample rate in Hz (default: 44100)
--audio-channels     Audio channels (default: 1)
--min-free-percent   Auto-stop if free disk % drops below threshold (default: 10.0)
--disk-check-seconds How often to check disk free space (default: 1.0)
```

Notes:

- `--stop-key` uses only the first character you provide.
- Camera drivers may not honor your exact requested width/height/FPS; actual values are used for output.
- Audio capture requires `sounddevice`.
- Muxing audio into final `.mp4` requires `ffmpeg`; without it, the app saves video-only `.mp4` plus a separate `.wav`.
- Recording auto-stops before disk exhaustion based on `--min-free-percent`.

## Examples

Basic run:

```bash
python3 privacy_cam_cli.py
```

Use key `x` to stop:

```bash
python3 privacy_cam_cli.py --stop-key x
```

Use external camera (often index `1`):

```bash
python3 privacy_cam_cli.py --camera-index 1
```

Higher resolution and custom FPS:

```bash
python3 privacy_cam_cli.py --width 1920 --height 1080 --fps 30
```

Save to a custom directory:

```bash
python3 privacy_cam_cli.py --output-dir ./evidence
```

Combined:

```bash
python3 privacy_cam_cli.py --camera-index 1 --width 1920 --height 1080 --fps 24 --stop-key s --output-dir ./evidence
```

Record with microphone audio:

```bash
python3 privacy_cam_cli.py --audio
```

Record with audio + custom audio settings:

```bash
python3 privacy_cam_cli.py --audio --audio-device 0 --audio-samplerate 48000 --audio-channels 2
```

Use a stricter disk safety margin (stop at 10% free):

```bash
python3 privacy_cam_cli.py --min-free-percent 10
```

Use a looser disk safety margin (stop at 5% free):

```bash
python3 privacy_cam_cli.py --min-free-percent 5
```

## Output Files

Saved file format:

```text
recordings/hotel_safety_cam_YYYYMMDD_HHMMSS_WIDTHxHEIGHT.mp4
```

At stop, the app prints:

- frame count
- elapsed time
- final saved path

## Event Scrubber Timeline Tool

Analyze a recorded MP4 for notable events (motion spikes, scene changes, people, and audio spikes) and generate:

- JSON event data
- a standalone HTML timeline report with clickable markers that seek to each event

Run:

```bash
python3 video_event_scrubber.py recordings/your_video.mp4
```

Output defaults to `scrub_reports/`.

Useful flags:

```text
--output-dir STR       Output folder for JSON + HTML reports
--sample-fps FLOAT     Visual analysis sample rate (default: 2.0)
--people-every-sec     Person detection interval in seconds (default: 1.0)
--motion-threshold     Motion spike threshold (default: 12.0)
--scene-threshold      Scene-change threshold (default: 0.22)
--skip-people          Disable person detection
--skip-audio           Disable audio spike detection
--max-seconds FLOAT    Analyze only first N seconds
```

Example:

```bash
python3 video_event_scrubber.py recordings/hotel_safety_cam_20260217_094941_1280x720.mp4 --sample-fps 3 --output-dir scrub_reports
```

## Privacy & Practical Use

- Position camera to cover room entry points.
- Verify visible status before leaving.
- Test a short recording first to confirm camera index and framing.
- Keep files in a secure folder and back up if needed.
