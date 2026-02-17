#!/usr/bin/env python3
"""Stitch segmented HotelSafetyCam MP4 parts into one output MP4."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PART_RE = re.compile(r"^(?P<stem>.+)_part(?P<idx>\d{4})\.mp4$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge segmented recording parts into a single MP4 using ffmpeg concat demuxer. "
            "Input can be a session manifest, first part file, any part file, or a session stem."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to session JSON, path to a part MP4, or session stem/prefix "
            "(e.g., recordings/hotel_safety_cam_20260217_113038_1280x720)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output merged MP4 path (default: <stem>_stitched.mp4 next to segments)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered segments and output path without running ffmpeg",
    )
    return parser.parse_args()


def normalize_manifest_path(raw_path: str, manifest_dir: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    parsed = Path(normalized)
    if parsed.is_absolute():
        return parsed

    candidates = [
        (manifest_dir / parsed),
        (Path.cwd() / parsed),
    ]
    if parsed.parts and parsed.parts[0] == manifest_dir.name:
        candidates.append(manifest_dir.parent / parsed)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def load_manifest_segments(manifest_path: Path) -> Tuple[List[Path], str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read manifest {manifest_path}: {exc}") from exc

    session = payload.get("session")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RuntimeError(f"Manifest has no segments: {manifest_path}")

    manifest_dir = manifest_path.parent
    parts: List[Path] = []
    for item in raw_segments:
        if not isinstance(item, dict) or "path" not in item:
            raise RuntimeError(f"Invalid segment entry in manifest: {item!r}")
        parts.append(normalize_manifest_path(str(item["path"]), manifest_dir))

    if not session or not isinstance(session, str):
        session = manifest_path.stem.replace("_session", "")

    return parts, session


def split_candidate(input_value: str) -> Tuple[Path, str]:
    candidate = Path(input_value)
    if candidate.suffix.lower() == ".mp4":
        match = PART_RE.match(candidate.name)
        if not match:
            raise RuntimeError(
                "When input is an MP4 file, it must match *_part####.mp4 naming."
            )
        return candidate.parent, match.group("stem")

    if candidate.name.endswith("_session"):
        return candidate.parent, candidate.name[: -len("_session")]

    return candidate.parent, candidate.name


def discover_segment_parts(input_value: str) -> Tuple[List[Path], str]:
    search_dir, stem = split_candidate(input_value)
    search_dir = search_dir if str(search_dir) else Path(".")

    part_paths: List[Tuple[int, Path]] = []
    for path in search_dir.glob(f"{stem}_part*.mp4"):
        match = PART_RE.match(path.name)
        if not match or match.group("stem") != stem:
            continue
        part_paths.append((int(match.group("idx")), path.resolve()))

    if not part_paths:
        raise RuntimeError(
            f"No segment parts found for stem '{stem}' in {search_dir.resolve()}"
        )

    part_paths.sort(key=lambda item: item[0])
    return [path for _, path in part_paths], stem


def resolve_input(input_value: str) -> Tuple[List[Path], str]:
    input_path = Path(input_value)
    if input_path.exists() and input_path.suffix.lower() == ".json":
        return load_manifest_segments(input_path.resolve())
    return discover_segment_parts(input_value)


def choose_output(parts: Sequence[Path], session_stem: str, output_flag: Optional[str]) -> Path:
    if output_flag:
        return Path(output_flag).resolve()
    return (parts[0].parent / f"{session_stem}_stitched.mp4").resolve()


def validate_parts(parts: Sequence[Path]) -> None:
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(f"Missing segment files:\n{joined}")


def run_ffmpeg_concat(parts: Sequence[Path], output_path: Path, overwrite: bool) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = Path(f.name)
        for part in parts:
            # ffmpeg concat demuxer expects shell-safe quoted file entries.
            escaped = str(part).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
        ]
        cmd.append("-y" if overwrite else "-n")
        cmd.append(str(output_path))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg concat failed. Verify all segments use matching codecs and parameters."
            )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    args = parse_args()

    try:
        parts, stem = resolve_input(args.input)
        validate_parts(parts)
        output_path = choose_output(parts, stem, args.output)

        print(f"Session stem: {stem}")
        print(f"Segments found: {len(parts)}")
        for part in parts:
            print(f"  - {part}")
        print(f"Output: {output_path}")

        if args.dry_run:
            return 0

        if output_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"Output already exists: {output_path} (use --overwrite to replace)"
            )

        run_ffmpeg_concat(parts, output_path, overwrite=args.overwrite)
        print("Stitch complete")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
