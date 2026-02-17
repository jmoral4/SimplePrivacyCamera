# Hotel Safety Cam (CLI)

A small privacy/safety camera recorder for hotel rooms or temporary spaces.

It records from your webcam, shows a clear terminal recording state, and lets you stop recording with a single key while safely writing the video to disk.

<img width="833" height="873" alt="image" src="https://github.com/user-attachments/assets/0342a9a1-df54-4db4-be5d-26aa8a07536c" />

## Features

- Clear status while active: blinking `[REC]/[rec]` in terminal.
- Clear status when stopped: `[IDLE]`.
- Keyboard stop key (default `q`) without pressing Enter.
- Also supports `Ctrl+C` to stop and save.
- Writes timestamped chunked `.mp4` segment files to disk.

## Requirements

- Python 3.8+
- Webcam/camera device
- `opencv-python`
- `onnxruntime` (recommended for ONNX model compatibility when OpenCV DNN import fails)
- Optional for audio capture: `sounddevice`
- Optional for per-segment audio/video muxing: `ffmpeg` on PATH

Install dependency:

```bash
python3 -m pip install opencv-python
```

Recommended for person-model inference fallback:

```bash
python3 -m pip install onnxruntime
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
--segment-max-mb     Rotate to a new segment at this size in MB (default: 950)
--segment-max-seconds Rotate to a new segment at this duration in seconds (default: 1200)
--writer-queue-seconds Approx queue depth in seconds awaiting writer thread (default: 5.0)
```

Notes:

- `--stop-key` uses only the first character you provide.
- Camera drivers may not honor your exact requested width/height/FPS; actual values are used for output.
- Audio capture requires `sounddevice`.
- Audio muxing is done per segment in the background and requires `ffmpeg`; without it, segments are saved video-only.
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

Approximate 1GB chunks or 20-minute chunks:

```bash
python3 privacy_cam_cli.py --segment-max-mb 950 --segment-max-seconds 1200
```

Use a 5-second writer queue (default):

```bash
python3 privacy_cam_cli.py --writer-queue-seconds 5
```

## Output Files

Saved segment format:

```text
recordings/hotel_safety_cam_YYYYMMDD_HHMMSS_WIDTHxHEIGHT_part0001.mp4
recordings/hotel_safety_cam_YYYYMMDD_HHMMSS_WIDTHxHEIGHT_part0002.mp4
...
recordings/hotel_safety_cam_YYYYMMDD_HHMMSS_WIDTHxHEIGHT_session.json
```

At stop, the app prints:

- frame count
- elapsed time
- number of saved segments
- session manifest path

## Stitch Segments Into One MP4

Use the stitch utility to merge segmented outputs into a single MP4 for downstream tooling.

```bash
python3 stitch_recording_segments.py recordings/hotel_safety_cam_YYYYMMDD_HHMMSS_WIDTHxHEIGHT_session.json
```

For your current run:

```bash
python3 stitch_recording_segments.py recordings/hotel_safety_cam_20260217_113038_1280x720_session.json
```

Useful flags:

```text
--output PATH   Custom output MP4 path
--overwrite     Replace existing output file
--dry-run       Show discovered parts without running ffmpeg
```

## Event Scrubber Timeline Tool

Analyze a recorded MP4 for notable events (motion spikes, scene changes, people, and audio spikes) and generate:

- JSON event data
- a standalone HTML timeline report with clickable markers that seek to each event

Person detection behavior:

- DNN (default): uses an ONNX model (`models/yolo11n.onnx`) for faster, more accurate modern object detection.
- HOG fallback: if the ONNX model/runtime cannot load, it falls back to OpenCV's classic Histogram of Oriented Gradients pedestrian detector.

Run:

```bash
python3 video_event_scrubber.py recordings/your_video.mp4
```

Output defaults to `scrub_reports/`.

Useful flags:

```text
--output-dir STR       Output folder for JSON + HTML reports
--regen-from-json PATH Regenerate HTML from an existing *_events_*.json
--regen-from-output-dir DIR  Regenerate HTML from latest *_events_*.json in directory
--html-video-src PATH  Override embedded video path/URL in regenerated HTML
--sample-fps FLOAT     Visual analysis sample rate (default: 2.0)
--people-every-sec     Person detection interval in seconds (default: 1.0)
--motion-threshold     Motion spike threshold (default: 12.0)
--scene-threshold      Scene-change threshold (default: 0.22)
--person-model PATH    ONNX detector path (default: models/yolo11n.onnx)
--person-conf-threshold FLOAT  Person confidence threshold (default: 0.20)
--person-nms-threshold FLOAT   Person NMS IoU threshold (default: 0.45)
--person-input-size INT        ONNX input size (default: 640)
--person-bottom-roi FLOAT      Bottom ROI fraction for extra pass (default: 0.45)
--person-roi-upscale FLOAT     Bottom ROI upscale factor (default: 1.6)
--require-people-model         Fail if ONNX model cannot load (no HOG fallback)
--skip-people          Disable person detection
--skip-audio           Disable audio spike detection
--max-seconds FLOAT    Analyze only first N seconds
```

Example:

```bash
python3 video_event_scrubber.py recordings/hotel_safety_cam_20260217_094941_1280x720.mp4 --sample-fps 3 --output-dir scrub_reports
```

Fast HTML-only regenerate (no re-analysis):

```bash
python3 video_event_scrubber.py --regen-from-output-dir scrub_reports
```

Or from a specific events JSON:

```bash
python3 video_event_scrubber.py --regen-from-json scrub_reports/your_video_events_YYYYMMDD_HHMMSS.json
```

The regenerated report now includes a dedicated **People Scrub Timeline** lane plus next/previous person jump controls.

## Privacy & Practical Use

- Position camera to cover room entry points.
- Verify visible status before leaving.
- Test a short recording first to confirm camera index and framing.
- Keep files in a secure folder and back up if needed.
