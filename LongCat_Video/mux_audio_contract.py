from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator

from .video_output import normalize_mux_audio_path, validate_mux_audio_path


MAX_NATIVE_AUDIO_SAMPLE_RATE = 384_000
NATIVE_MUX_DIRECTORY = "longcat_avatar_mux"


@dataclass(frozen=True)
class ValidatedNativeAudio:
    waveform: Any
    sample_rate: int
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class MuxAudioSource:
    kind: str
    native_audio: Mapping[str, Any] | None = None
    legacy_path: str = ""

    @property
    def enabled(self) -> bool:
        return self.kind != "disabled"


def validate_native_audio_payload(audio: Any) -> ValidatedNativeAudio:
    if not isinstance(audio, Mapping):
        raise TypeError("mux_audio must be a native AUDIO mapping.")
    if "waveform" not in audio:
        raise ValueError("mux_audio is missing its waveform.")
    if "sample_rate" not in audio:
        raise ValueError("mux_audio is missing its sample_rate.")

    waveform = audio["waveform"]
    if not all(hasattr(waveform, name) for name in ("shape", "isfinite", "detach")):
        raise TypeError("mux_audio waveform must be a tensor.")
    try:
        shape = tuple(int(value) for value in waveform.shape)
    except (TypeError, ValueError) as exc:
        raise TypeError("mux_audio waveform must expose a valid tensor shape.") from exc
    if len(shape) != 3:
        raise ValueError("mux_audio waveform must have batch, channels, samples dimensions.")
    batch, channels, samples = shape
    if batch != 1:
        raise ValueError("mux_audio supports exactly one batch item for one video output.")
    if channels not in (1, 2):
        raise ValueError("mux_audio must contain one or two channels.")
    if samples < 1:
        raise ValueError("mux_audio must contain one or more samples.")

    sample_rate = audio["sample_rate"]
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("mux_audio sample_rate must be an integer.")
    if sample_rate < 1 or sample_rate > MAX_NATIVE_AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"mux_audio sample_rate must be between 1 and {MAX_NATIVE_AUDIO_SAMPLE_RATE}."
        )

    try:
        is_finite = bool(waveform.isfinite().all().item())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise TypeError("mux_audio waveform must support finite tensor validation.") from exc
    if not is_finite:
        raise ValueError("mux_audio waveform samples must be finite.")

    return ValidatedNativeAudio(waveform=waveform, sample_rate=sample_rate, shape=shape)


def select_mux_audio_source(
    native_audio: Any,
    legacy_path: str | os.PathLike[str] | None,
    *,
    allowed_roots: Sequence[str | os.PathLike[str]],
    opener: Callable[..., Any] = open,
) -> MuxAudioSource:
    normalized_legacy = normalize_mux_audio_path(legacy_path)
    if native_audio is not None and normalized_legacy:
        raise ValueError("Connect either native mux_audio or legacy mux_audio_path, not both.")
    if native_audio is not None:
        validate_native_audio_payload(native_audio)
        return MuxAudioSource(kind="native", native_audio=native_audio)
    if normalized_legacy:
        validated_path = validate_mux_audio_path(
            normalized_legacy,
            allowed_roots=allowed_roots,
            opener=opener,
        )
        return MuxAudioSource(kind="legacy", legacy_path=validated_path)
    return MuxAudioSource(kind="disabled")


def _default_writer(path: str, samples: Any, sample_rate: int, **kwargs: Any) -> None:
    # CRITICAL: keep this import lazy so schema discovery does not acquire an audio codec dependency.
    import soundfile

    soundfile.write(path, samples, sample_rate, **kwargs)


def _resolved_temp_directory(temp_dir: str | os.PathLike[str]) -> Path:
    if not temp_dir:
        raise ValueError("The ComfyUI temp directory is required for native mux audio.")
    try:
        temp_root = Path(temp_dir).expanduser().resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        private_root = temp_root / NATIVE_MUX_DIRECTORY
        private_root.mkdir(parents=True, exist_ok=True)
        resolved_private = private_root.resolve()
    except (OSError, RuntimeError):
        raise PermissionError("The ComfyUI temp directory is unavailable for native mux audio.") from None
    try:
        resolved_private.relative_to(temp_root)
    except ValueError:
        raise ValueError("Native mux audio temp directory escaped the ComfyUI temp directory.") from None
    return resolved_private


@contextmanager
def materialized_mux_audio(
    source: MuxAudioSource,
    *,
    temp_dir: str | os.PathLike[str],
    writer: Callable[..., None] | None = None,
) -> Iterator[str]:
    if source.kind == "disabled":
        yield ""
        return
    if source.kind == "legacy":
        yield source.legacy_path
        return
    if source.kind != "native" or source.native_audio is None:
        raise ValueError("Unsupported mux audio source selection.")

    validated = validate_native_audio_payload(source.native_audio)
    private_root = _resolved_temp_directory(temp_dir)
    try:
        file_descriptor, materialized_path = tempfile.mkstemp(
            prefix="mux_",
            suffix=".wav",
            dir=private_root,
        )
    except OSError:
        raise PermissionError("A native mux audio temp file could not be created.") from None
    os.close(file_descriptor)
    path = Path(materialized_path)
    try:
        try:
            path.resolve().relative_to(private_root)
        except ValueError:
            raise ValueError("Native mux audio path escaped the ComfyUI temp directory.") from None
        samples = validated.waveform.detach().cpu().float().squeeze(0).transpose(0, 1).numpy()
        try:
            (writer or _default_writer)(
                str(path),
                samples,
                validated.sample_rate,
                format="WAV",
                subtype="FLOAT",
            )
        except Exception:
            raise RuntimeError("Native mux audio materialization failed.") from None
        yield str(path)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            raise RuntimeError("Native mux audio temp cleanup failed.") from None
