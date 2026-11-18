from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable, Mapping, Sequence

from .official_asset_provenance import (
    AVATAR_INT8_SHARDED_PROVENANCE,
    AVATAR_SHARDED_PROVENANCE,
    TEXT_ENCODER_PROVENANCE,
    OFFICIAL_ASSET_PROVENANCE,
    OfficialAssetProvenance,
)


AVATAR_V15 = "avatar-v1.5"
SINGLE_FILE_SAFETENSORS = "single_file_safetensors"
OFFICIAL_SHARDED = "official_sharded"
OFFICIAL_INT8_SHARDED = "official_int8_sharded"
SUPPORTED_CHECKPOINT_SOURCES = (
    SINGLE_FILE_SAFETENSORS,
    OFFICIAL_SHARDED,
    OFFICIAL_INT8_SHARDED,
)
OFFICIAL_CHECKPOINT_SOURCES = (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED)
SINGLE_FILE_CHECKPOINT_SOURCES = (SINGLE_FILE_SAFETENSORS,)

OFFICIAL_AVATAR_REPO_ID = AVATAR_SHARDED_PROVENANCE.repo_id
OFFICIAL_AVATAR_REVISION = AVATAR_SHARDED_PROVENANCE.revision
OFFICIAL_AVATAR_MODEL_DIR = "LongCat-Video-Avatar-1.5"
OFFICIAL_BASE_REPO_ID = TEXT_ENCODER_PROVENANCE.repo_id
OFFICIAL_BASE_REVISION = TEXT_ENCODER_PROVENANCE.revision
OFFICIAL_BASE_MODEL_DIR = "LongCat-Video"


def safe_display_path(path: str | os.PathLike[str] | None) -> str:
    if path is None:
        return "<not selected>"
    text = os.fspath(path)
    if not text or text == "none":
        return "<not selected>"
    normalized = os.path.normpath(text)
    name = os.path.basename(normalized)
    return name or normalized


def is_official_checkpoint_source(source_kind: str) -> bool:
    return source_kind in OFFICIAL_CHECKPOINT_SOURCES


def is_single_file_checkpoint_source(source_kind: str) -> bool:
    return source_kind in SINGLE_FILE_CHECKPOINT_SOURCES


def describe_checkpoint_source_role(source_kind: str) -> str:
    if is_official_checkpoint_source(source_kind):
        return "official_sharded"
    if is_single_file_checkpoint_source(source_kind):
        return "local_or_community_converted_single_file"
    raise ValueError(f"Unsupported checkpoint source: {source_kind}")


@dataclass(frozen=True)
class CheckpointSourceSpec:
    source_kind: str
    checkpoint_root: str
    model_path: str | None
    subfolder: str | None
    config_path: str | None
    quantization_config_path: str | None
    index_path: str | None
    index_name: str | None
    shard_paths: tuple[str, ...]
    shard_names: tuple[str, ...]
    required_files: tuple[str, ...]
    manifest_id: str | None
    use_int8: bool
    total_size: int | None = None


@dataclass(frozen=True)
class CheckpointInspection:
    source_kind: str
    checkpoint_root: str
    is_complete: bool
    missing_files: tuple[str, ...]
    spec: CheckpointSourceSpec | None = None


@dataclass(frozen=True)
class DownloadManifest:
    source_kind: str
    repo_id: str
    revision: str
    model_dir_name: str
    local_dir: str
    allow_patterns: tuple[str, ...]
    models_root: str
    provenance: OfficialAssetProvenance


@dataclass(frozen=True)
class DownloadResult:
    manifest: DownloadManifest
    local_dir: str


def _layout_for_source(source_kind: str) -> tuple[str, str, bool]:
    if source_kind == OFFICIAL_SHARDED:
        return "base_model", "diffusion_pytorch_model.safetensors.index.json", False
    if source_kind == OFFICIAL_INT8_SHARDED:
        return "base_model_int8", "quantized_model.safetensors.index.json", True
    raise ValueError(f"Unsupported official checkpoint source: {source_kind}")


def _safe_join(root: str, relative_path: str, label: str) -> str:
    if not relative_path:
        raise ValueError(f"Unsafe {label}: empty path")
    if os.path.isabs(relative_path):
        raise ValueError(f"Unsafe {label}: absolute paths are not allowed")
    normalized = os.path.normpath(relative_path)
    if normalized == "." or normalized.startswith("..") or os.path.isabs(normalized):
        raise ValueError(f"Unsafe {label}: {relative_path}")
    full_path = os.path.abspath(os.path.join(root, normalized))
    root_real = os.path.normcase(os.path.realpath(root))
    full_real = os.path.normcase(os.path.realpath(full_path))
    try:
        within_root = os.path.commonpath((root_real, full_real)) == root_real
    except ValueError:
        within_root = False
    if not within_root:
        raise ValueError(f"Unsafe {label}: path escapes checkpoint directory")
    return full_path


def _safe_shard_path(shard_dir: str, shard_reference: object) -> tuple[str, str]:
    if not isinstance(shard_reference, str):
        raise ValueError("Unsafe shard reference: shard filename must be a string")
    shard_path = _safe_join(shard_dir, shard_reference, "shard reference")
    if os.path.splitext(shard_path)[1].lower() != ".safetensors":
        raise ValueError(
            f"Unsupported shard file for {safe_display_path(shard_reference)}; "
            "expected .safetensors."
        )
    if os.path.isdir(shard_path):
        raise ValueError(f"Unsafe shard reference: {safe_display_path(shard_reference)} is a directory")
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Missing checkpoint shard: {safe_display_path(shard_path)}")
    return os.path.relpath(shard_path, shard_dir), shard_path


def _read_index(index_path: str) -> Mapping[str, object]:
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed sharded checkpoint index: {safe_display_path(index_path)}") from exc

    if not isinstance(data, dict):
        raise ValueError("Malformed sharded checkpoint index: expected JSON object")
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Malformed sharded checkpoint index: missing non-empty weight_map")
    return data


def _metadata_total_size(index_data: Mapping[str, object]) -> int | None:
    metadata = index_data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("total_size")
    if isinstance(value, int):
        return value
    return None


def _required_relative_files(source_kind: str) -> tuple[str, ...]:
    subfolder, index_name, use_int8 = _layout_for_source(source_kind)
    required = [
        f"{subfolder}/config.json",
        f"{subfolder}/{index_name}",
    ]
    if use_int8:
        required.insert(1, f"{subfolder}/quantization_config.json")
    return tuple(required)


def _coerce_checkpoint_root(source_kind: str, path: str) -> str:
    subfolder, index_name, _ = _layout_for_source(source_kind)
    absolute = os.path.abspath(path)
    if os.path.isfile(absolute) and os.path.basename(absolute) == index_name:
        parent = os.path.basename(os.path.dirname(absolute))
        if parent != subfolder:
            raise ValueError(
                f"Unsupported checkpoint index location for {safe_display_path(path)}; "
                f"expected {subfolder}/{index_name}."
            )
        return os.path.dirname(os.path.dirname(absolute))
    return absolute


def inspect_checkpoint_source(
    source_kind: str,
    path: str,
    *,
    model_type: str = AVATAR_V15,
) -> CheckpointInspection:
    if source_kind == SINGLE_FILE_SAFETENSORS:
        missing = () if path and os.path.isfile(path) else (safe_display_path(path),)
        if missing:
            return CheckpointInspection(source_kind, os.path.abspath(path or ""), False, missing)
        return CheckpointInspection(
            source_kind,
            os.path.abspath(os.path.dirname(path)),
            True,
            (),
            validate_checkpoint_source(source_kind, path, model_type=model_type),
        )

    if model_type != AVATAR_V15:
        raise ValueError("Official sharded checkpoints are only supported for Avatar v1.5.")

    checkpoint_root = _coerce_checkpoint_root(source_kind, path)
    missing = tuple(
        relative
        for relative in _required_relative_files(source_kind)
        if not os.path.isfile(os.path.join(checkpoint_root, relative))
    )
    if missing:
        return CheckpointInspection(source_kind, checkpoint_root, False, missing)

    try:
        spec = validate_checkpoint_source(source_kind, checkpoint_root, model_type=model_type)
    except FileNotFoundError as exc:
        message = str(exc).split(":", 1)[-1].strip()
        return CheckpointInspection(source_kind, checkpoint_root, False, (message,))
    return CheckpointInspection(source_kind, checkpoint_root, True, (), spec)


def validate_checkpoint_source(
    source_kind: str,
    path: str | None,
    *,
    model_type: str = AVATAR_V15,
) -> CheckpointSourceSpec:
    if source_kind == SINGLE_FILE_SAFETENSORS:
        if not path:
            raise FileNotFoundError("Missing DiT model: no diffusion model was selected.")
        if os.path.splitext(path)[1].lower() != ".safetensors":
            raise ValueError(
                f"Unsupported DiT model format for {safe_display_path(path)}. "
                "Expected .safetensors."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing DiT model: {safe_display_path(path)}")
        return CheckpointSourceSpec(
            source_kind=source_kind,
            checkpoint_root=os.path.abspath(os.path.dirname(path)),
            model_path=path,
            subfolder=None,
            config_path=None,
            quantization_config_path=None,
            index_path=None,
            index_name=None,
            shard_paths=(),
            shard_names=(),
            required_files=(os.path.basename(path),),
            manifest_id=None,
            use_int8="int8" in os.path.basename(path).lower(),
        )

    if source_kind not in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED):
        raise ValueError(f"Unsupported checkpoint source: {source_kind}")
    if model_type != AVATAR_V15:
        raise ValueError("Official sharded checkpoints are only supported for Avatar v1.5.")
    if not path:
        raise FileNotFoundError("Missing official checkpoint: no checkpoint root was selected.")

    checkpoint_root = _coerce_checkpoint_root(source_kind, path)
    subfolder, index_name, use_int8 = _layout_for_source(source_kind)
    shard_dir = os.path.join(checkpoint_root, subfolder)
    required_files = _required_relative_files(source_kind)
    for relative in required_files:
        full_path = os.path.join(checkpoint_root, relative)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Missing checkpoint file: {relative}")

    index_path = os.path.join(shard_dir, index_name)
    index_data = _read_index(index_path)
    weight_map = index_data["weight_map"]
    if not isinstance(weight_map, dict):
        raise ValueError("Malformed sharded checkpoint index: missing non-empty weight_map")

    shard_names: set[str] = set()
    shard_paths: set[str] = set()
    for shard_reference in weight_map.values():
        shard_name, shard_path = _safe_shard_path(shard_dir, shard_reference)
        shard_names.add(shard_name)
        shard_paths.add(shard_path)

    return CheckpointSourceSpec(
        source_kind=source_kind,
        checkpoint_root=checkpoint_root,
        model_path=None,
        subfolder=subfolder,
        config_path=os.path.join(shard_dir, "config.json"),
        quantization_config_path=(
            os.path.join(shard_dir, "quantization_config.json") if use_int8 else None
        ),
        index_path=index_path,
        index_name=index_name,
        shard_paths=tuple(sorted(shard_paths)),
        shard_names=tuple(sorted(shard_names)),
        required_files=required_files + tuple(f"{subfolder}/{name}" for name in sorted(shard_names)),
        manifest_id=source_kind,
        use_int8=use_int8,
        total_size=_metadata_total_size(index_data),
    )


def _ensure_download_target(models_dir: str, model_dir_name: str) -> str:
    if not model_dir_name:
        raise ValueError("Unsafe download target: empty model directory name")
    target = _safe_join(os.path.abspath(models_dir), model_dir_name, "download target")
    os.makedirs(os.path.abspath(models_dir), exist_ok=True)
    return target


def _provenance_for_source(source_kind: str) -> OfficialAssetProvenance:
    if source_kind == OFFICIAL_SHARDED:
        return AVATAR_SHARDED_PROVENANCE
    if source_kind == OFFICIAL_INT8_SHARDED:
        return AVATAR_INT8_SHARDED_PROVENANCE
    raise ValueError(f"No download manifest is defined for checkpoint source: {source_kind}")


def build_download_manifest(
    source_kind: str,
    models_dir: str,
    *,
    model_dir_name: str = OFFICIAL_AVATAR_MODEL_DIR,
) -> DownloadManifest:
    provenance = _provenance_for_source(source_kind)
    models_root = os.path.abspath(models_dir)

    return DownloadManifest(
        source_kind=source_kind,
        repo_id=provenance.repo_id,
        revision=provenance.revision,
        model_dir_name=model_dir_name,
        local_dir=_ensure_download_target(models_dir, model_dir_name),
        allow_patterns=provenance.allow_patterns,
        models_root=models_root,
        provenance=provenance,
    )


def build_text_encoder_download_manifest(
    models_dir: str,
    *,
    model_dir_name: str = OFFICIAL_BASE_MODEL_DIR,
) -> DownloadManifest:
    provenance = TEXT_ENCODER_PROVENANCE
    models_root = os.path.abspath(models_dir)
    return DownloadManifest(
        source_kind="official_longcat_text_encoder",
        repo_id=provenance.repo_id,
        revision=provenance.revision,
        model_dir_name=model_dir_name,
        local_dir=_ensure_download_target(models_dir, model_dir_name),
        allow_patterns=provenance.allow_patterns,
        models_root=models_root,
        provenance=provenance,
    )


def _import_snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic LongCat weight downloads. "
            "Install huggingface_hub or download the official checkpoint manually."
        ) from exc
    return snapshot_download


def download_missing_checkpoint_assets(
    manifest: DownloadManifest,
    *,
    snapshot_download: Callable[..., str] | None = None,
    local_files_only: bool = False,
) -> DownloadResult:
    _validate_download_manifest(manifest)
    downloader = snapshot_download or _import_snapshot_download()
    local_dir = downloader(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        allow_patterns=manifest.allow_patterns,
        local_dir=manifest.local_dir,
        local_files_only=local_files_only,
    )
    _validate_download_result(manifest, local_dir)
    return DownloadResult(manifest=manifest, local_dir=manifest.local_dir)


def _same_real_path(left: str, right: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))
    except (OSError, TypeError, ValueError):
        return False


def _validate_download_manifest(manifest: DownloadManifest) -> None:
    expected = OFFICIAL_ASSET_PROVENANCE.get(manifest.source_kind)
    if (
        expected is None
        or manifest.provenance != expected
        or manifest.repo_id != expected.repo_id
        or manifest.revision != expected.revision
        or manifest.allow_patterns != expected.allow_patterns
    ):
        raise ValueError("Unapproved official asset download manifest")

    approved_target = _safe_join(
        os.path.abspath(manifest.models_root),
        manifest.model_dir_name,
        "download target",
    )
    if not _same_real_path(approved_target, manifest.local_dir):
        raise ValueError("Unsafe download target: manifest target does not match the approved model root")


def _validate_download_result(manifest: DownloadManifest, local_dir: object) -> None:
    if not isinstance(local_dir, (str, os.PathLike)) or not _same_real_path(
        manifest.local_dir,
        os.fspath(local_dir),
    ):
        raise ValueError("Unsafe download result: downloader returned an unapproved directory")

    # CRITICAL: re-check after download; a downloader must not replace the approved target with an escaping symlink.
    try:
        relative_target = os.path.relpath(manifest.local_dir, manifest.models_root)
        _safe_join(manifest.models_root, relative_target, "download result")
    except (OSError, ValueError) as exc:
        raise ValueError("Unsafe download result: target escapes the approved model root") from exc

    for entry in manifest.provenance.content:
        required_path = _safe_join(manifest.local_dir, entry.path, "downloaded asset")
        if not os.path.isfile(required_path):
            raise FileNotFoundError(f"Official asset download is incomplete: {entry.path}")
        if os.path.getsize(required_path) != entry.size:
            raise ValueError(f"Official asset download has an invalid size: {entry.path}")

    if manifest.source_kind in OFFICIAL_CHECKPOINT_SOURCES:
        validate_checkpoint_source(manifest.source_kind, manifest.local_dir)
        return

    if manifest.source_kind == TEXT_ENCODER_PROVENANCE.asset_role:
        index_path = _safe_join(
            manifest.local_dir,
            "text_encoder/model.safetensors.index.json",
            "text encoder index",
        )
        index_data = _read_index(index_path)
        shard_dir = os.path.dirname(index_path)
        for shard_reference in index_data["weight_map"].values():
            _safe_shard_path(shard_dir, shard_reference)
        return

    raise ValueError(f"Unsupported download manifest role: {manifest.source_kind}")
