from __future__ import annotations

from dataclasses import dataclass
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable


DEFAULT_VIDEO_FPS = 25
DEFAULT_VIDEO_QUALITY = 5
DEFAULT_VIDEO_PREFIX = "longcat_avatar"
DISABLED_MUX_AUDIO_SENTINELS = {"", "0", "none", "null"}
SUPPORTED_MUX_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".wav"}


@dataclass(frozen=True)
class VideoOutputPlan:
    enabled: bool
    stem_path: str
    final_path: str


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def sanitize_video_prefix(prefix: str | None) -> str:
    prefix = prefix or DEFAULT_VIDEO_PREFIX
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._-")
    return sanitized or DEFAULT_VIDEO_PREFIX


def validate_output_directory(output_dir: str | os.PathLike[str]) -> Path:
    if not output_dir:
        raise ValueError("output_dir is required for video output.")
    output_root = Path(output_dir).expanduser().resolve()
    if "reference" in {part.lower() for part in output_root.parts}:
        raise ValueError("Video output directory must not be inside reference.")
    return output_root


def normalize_mux_audio_path(audio_path: str | os.PathLike[str] | None) -> str:
    if audio_path is None:
        return ""
    normalized = os.fspath(audio_path).strip()
    # IMPORTANT: legacy workflows may serialize disabled String inputs as "0"; keep that as mux disabled.
    if normalized.lower() in DISABLED_MUX_AUDIO_SENTINELS:
        return ""
    return normalized


def validate_mux_audio_path(
    audio_path: str | os.PathLike[str] | None,
    *,
    allowed_roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...] | None = None,
    opener: Callable[..., Any] = open,
) -> str:
    audio_path = normalize_mux_audio_path(audio_path)
    if not audio_path:
        raise ValueError("mux_audio_path is required when video output is enabled.")
    if audio_path.startswith(("\\\\", "//")):
        raise ValueError("UNC mux audio paths are not allowed.")
    try:
        resolved_path = Path(audio_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileNotFoundError("mux_audio_path does not point to a local audio file.")
    if not resolved_path.is_file():
        raise FileNotFoundError("mux_audio_path does not point to a local audio file.")
    if resolved_path.suffix.lower() not in SUPPORTED_MUX_AUDIO_EXTENSIONS:
        raise ValueError("mux_audio_path must use a supported audio media extension.")
    if allowed_roots is not None:
        resolved_roots = [Path(root).expanduser().resolve() for root in allowed_roots if root]
        if not any(_is_relative_to(resolved_path, root) for root in resolved_roots):
            raise ValueError("mux_audio_path must stay inside a ComfyUI host input, output, or temp directory.")
    try:
        with opener(resolved_path, "rb") as handle:
            handle.read(1)
    except (OSError, PermissionError) as exc:
        raise PermissionError("mux_audio_path is not readable.") from exc
    return str(resolved_path)


def build_video_output_plan(
    *,
    enabled: bool,
    output_dir: str | os.PathLike[str],
    audio_path: str | os.PathLike[str] | None,
    prefix: str | None = DEFAULT_VIDEO_PREFIX,
    mode: str = "single",
    token: str | None = None,
) -> VideoOutputPlan:
    if not enabled:
        return VideoOutputPlan(enabled=False, stem_path="", final_path="")

    validate_mux_audio_path(audio_path)
    output_root = validate_output_directory(output_dir)
    token = token or uuid.uuid4().hex[:10]
    safe_prefix = sanitize_video_prefix(prefix)
    safe_mode = sanitize_video_prefix(mode)
    stem = output_root / f"{safe_prefix}_{safe_mode}_{token}"
    final_path = stem.with_suffix(".mp4")
    if not _is_relative_to(final_path.resolve().parent, output_root):
        raise ValueError("Video output path escaped the ComfyUI output directory.")
    return VideoOutputPlan(enabled=True, stem_path=str(stem), final_path=str(final_path))


def save_muxed_video(
    frames: Any,
    *,
    enabled: bool,
    output_dir: str | os.PathLike[str],
    audio_path: str | os.PathLike[str] | None,
    prefix: str | None,
    mode: str,
    fps: int = DEFAULT_VIDEO_FPS,
    quality: int = DEFAULT_VIDEO_QUALITY,
    saver: Callable[..., None] | None = None,
) -> str:
    plan = build_video_output_plan(
        enabled=enabled,
        output_dir=output_dir,
        audio_path=audio_path,
        prefix=prefix,
        mode=mode,
    )
    if not plan.enabled:
        return ""

    os.makedirs(os.path.dirname(plan.stem_path), exist_ok=True)
    if saver is None:
        from .longcat_video.audio_process.torch_utils import save_video_ffmpeg

        saver = save_video_ffmpeg
    saver(frames, plan.stem_path, os.fspath(audio_path), fps=fps, quality=quality)
    return plan.final_path
