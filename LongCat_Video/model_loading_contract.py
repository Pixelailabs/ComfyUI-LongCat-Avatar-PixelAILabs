from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
)


@dataclass(frozen=True)
class DitLoadPlan:
    loader_kind: str
    checkpoint_root: str | None
    subfolder: str | None
    index_path: str | None
    single_file: str | None
    shard_paths: tuple[str, ...]
    use_int8: bool


def resolve_dit_load_plan(contract: object) -> DitLoadPlan:
    source_kind = getattr(contract, "source_kind", SINGLE_FILE_SAFETENSORS)
    use_int8 = bool(getattr(contract, "use_int8", False))

    if source_kind == SINGLE_FILE_SAFETENSORS:
        model_path = getattr(contract, "model_path", None)
        return DitLoadPlan(
            loader_kind="single_file_int8" if use_int8 else "single_file_safetensors",
            checkpoint_root=None,
            subfolder=None,
            index_path=None,
            single_file=model_path,
            shard_paths=(),
            use_int8=use_int8,
        )

    if source_kind == OFFICIAL_SHARDED:
        return DitLoadPlan(
            loader_kind="official_sharded",
            checkpoint_root=getattr(contract, "checkpoint_root", None),
            subfolder=getattr(contract, "checkpoint_subfolder", "base_model"),
            index_path=getattr(contract, "checkpoint_index_path", None),
            single_file=None,
            shard_paths=tuple(getattr(contract, "checkpoint_shard_paths", ()) or ()),
            use_int8=False,
        )

    if source_kind == OFFICIAL_INT8_SHARDED:
        return DitLoadPlan(
            loader_kind="official_sharded_int8",
            checkpoint_root=getattr(contract, "checkpoint_root", None),
            subfolder=getattr(contract, "checkpoint_subfolder", "base_model_int8"),
            index_path=getattr(contract, "checkpoint_index_path", None),
            single_file=None,
            shard_paths=tuple(getattr(contract, "checkpoint_shard_paths", ()) or ()),
            use_int8=True,
        )

    raise ValueError(f"Unsupported DiT load source: {source_kind}")


def _safe_name(path: object) -> str:
    if not isinstance(path, (str, os.PathLike)):
        return "checkpoint"
    name = os.path.basename(os.fspath(path))
    return name if len(name) <= 160 and re.fullmatch(r"[A-Za-z0-9_.-]+", name) else "<invalid checkpoint name>"


def _safe_key(key: str) -> str:
    return key if len(key) <= 160 and re.fullmatch(r"[A-Za-z0-9_.-]+", key) else "<invalid tensor key>"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Malformed sharded checkpoint index: duplicate JSON key")
        result[key] = value
    return result


def _read_index(index_path: str) -> tuple[int, Mapping[str, str]]:
    if not isinstance(index_path, str) or not index_path:
        raise ValueError("Sharded checkpoint index path is required.")
    if not index_path.lower().endswith(".safetensors.index.json"):
        raise ValueError("Unsupported checkpoint index; expected .safetensors.index.json.")
    try:
        with open(index_path, encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except ValueError as exc:
        if "duplicate JSON key" in str(exc):
            raise
        raise ValueError(f"Malformed sharded checkpoint index: {_safe_name(index_path)}") from None
    except OSError:
        raise ValueError(f"Unable to read sharded checkpoint index: {_safe_name(index_path)}") from None

    if not isinstance(data, Mapping):
        raise ValueError("Malformed sharded checkpoint index: expected JSON object")
    metadata = data.get("metadata")
    weight_map = data.get("weight_map")
    total_size = metadata.get("total_size") if isinstance(metadata, Mapping) else None
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size <= 0:
        raise ValueError("Malformed sharded checkpoint index: invalid metadata.total_size")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Malformed sharded checkpoint index: missing non-empty weight_map")
    for key, shard_reference in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(shard_reference, str) or not shard_reference:
            raise ValueError("Malformed sharded checkpoint index: invalid weight_map entry")
    return total_size, weight_map


def _resolve_index_shards(index_path: str, weight_map: Mapping[str, str]) -> dict[str, set[str]]:
    index_directory = os.path.realpath(os.path.dirname(index_path))
    expected: dict[str, set[str]] = {}
    for key, shard_reference in weight_map.items():
        # SECURITY: shard references are untrusted index data; keep every resolved file under the index directory.
        if os.path.isabs(shard_reference):
            raise ValueError("Unsafe checkpoint shard reference")
        shard_path = os.path.realpath(os.path.join(index_directory, shard_reference))
        try:
            inside_index_directory = os.path.normcase(
                os.path.commonpath((index_directory, shard_path))
            ) == os.path.normcase(index_directory)
        except ValueError:
            inside_index_directory = False
        if not inside_index_directory:
            raise ValueError("Unsafe checkpoint shard reference")
        if os.path.splitext(shard_path)[1].lower() != ".safetensors":
            raise ValueError(f"Unsupported checkpoint shard: {_safe_name(shard_reference)}")
        expected.setdefault(os.path.normcase(shard_path), set()).add(key)
    return expected


def _validate_provided_shards(
    shard_paths: Sequence[str],
    expected: Mapping[str, set[str]],
) -> dict[str, str]:
    provided: dict[str, str] = {}
    for shard_path in shard_paths:
        resolved = os.path.realpath(shard_path)
        normalized = os.path.normcase(resolved)
        if normalized in provided:
            raise ValueError(f"Duplicate checkpoint shard path: {_safe_name(shard_path)}")
        if os.path.splitext(resolved)[1].lower() != ".safetensors":
            raise ValueError(f"Unsupported checkpoint shard: {_safe_name(shard_path)}")
        provided[normalized] = resolved

    missing = sorted(expected.keys() - provided.keys())
    unexpected = sorted(provided.keys() - expected.keys())
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing shards: " + ", ".join(_safe_name(path) for path in missing[:10]))
        if unexpected:
            details.append("unexpected shards: " + ", ".join(_safe_name(path) for path in unexpected[:10]))
        raise ValueError("Sharded checkpoint shard set mismatch; " + "; ".join(details))
    return provided


def _tensor_nbytes(
    key: str,
    tensor: object,
    allowed_dtypes: frozenset[str],
    expected_tensor: object | None,
) -> int:
    dtype = getattr(tensor, "dtype", None)
    shape = getattr(tensor, "shape", None)
    nbytes = getattr(tensor, "nbytes", None)
    try:
        normalized_shape = tuple(shape)
    except TypeError as exc:
        raise ValueError(f"invalid tensor metadata for {_safe_key(key)}") from exc
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in normalized_shape):
        raise ValueError(f"invalid tensor metadata for {_safe_key(key)}")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ValueError(f"invalid tensor metadata for {_safe_key(key)}")
    if str(dtype) not in allowed_dtypes:
        raise ValueError(f"Checkpoint tensor {_safe_key(key)} has unsupported dtype {dtype}")
    if expected_tensor is not None and tuple(getattr(expected_tensor, "shape", ())) != normalized_shape:
        raise ValueError(f"Checkpoint tensor {_safe_key(key)} has incompatible shape")
    return nbytes


def load_indexed_safetensors_state_dict(
    index_path: str,
    shard_paths: Sequence[str],
    *,
    allowed_dtypes: Sequence[str],
    expected_state: Mapping[str, object] | None = None,
    load_file: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate a complete indexed mapping before the caller's single assignment boundary."""

    total_size, weight_map = _read_index(index_path)
    expected = _resolve_index_shards(index_path, weight_map)
    provided = _validate_provided_shards(shard_paths, expected)
    if isinstance(allowed_dtypes, (str, bytes)):
        raise ValueError("Checkpoint tensor dtypes must be a sequence of dtype names.")
    normalized_dtypes = frozenset(allowed_dtypes)
    if not normalized_dtypes or any(not isinstance(dtype, str) or not dtype for dtype in normalized_dtypes):
        raise ValueError("At least one checkpoint tensor dtype must be allowed.")
    if load_file is None:
        from safetensors.torch import load_file as safetensors_load_file

        load_file = safetensors_load_file

    if expected_state is not None:
        expected_model_keys = set(expected_state)
        indexed_keys = set(weight_map)
        missing_model_keys = sorted(expected_model_keys - indexed_keys)
        unexpected_model_keys = sorted(indexed_keys - expected_model_keys)
        if missing_model_keys or unexpected_model_keys:
            details: list[str] = []
            if missing_model_keys:
                details.append(
                    "missing keys: " + ", ".join(_safe_key(key) for key in missing_model_keys[:10])
                )
            if unexpected_model_keys:
                details.append(
                    "unexpected keys: " + ", ".join(_safe_key(key) for key in unexpected_model_keys[:10])
                )
            raise ValueError("Checkpoint state dict mismatch; " + "; ".join(details))

    state_dict: dict[str, object] = {}
    loaded_bytes = 0
    for normalized_path, expected_keys in expected.items():
        shard_path = provided[normalized_path]
        try:
            shard_state = load_file(shard_path, device="cpu")
        except Exception:
            raise ValueError(f"Failed to load checkpoint shard {_safe_name(shard_path)}") from None
        if not isinstance(shard_state, Mapping):
            raise ValueError(f"Checkpoint shard {_safe_name(shard_path)} did not return a state mapping")
        actual_keys = set(shard_state)
        if any(not isinstance(key, str) or not key for key in actual_keys):
            raise ValueError(f"Checkpoint shard {_safe_name(shard_path)} has invalid tensor keys")
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append("missing keys: " + ", ".join(_safe_key(key) for key in missing[:10]))
            if unexpected:
                details.append("unexpected keys: " + ", ".join(_safe_key(key) for key in unexpected[:10]))
            raise ValueError(f"Shard key mismatch for {_safe_name(shard_path)}; " + "; ".join(details))
        duplicates = sorted(actual_keys & state_dict.keys())
        if duplicates:
            raise ValueError(
                "Duplicate checkpoint tensor keys: " + ", ".join(_safe_key(key) for key in duplicates[:10])
            )
        for key, tensor in shard_state.items():
            expected_tensor = expected_state.get(key) if expected_state is not None else None
            loaded_bytes += _tensor_nbytes(key, tensor, normalized_dtypes, expected_tensor)
            state_dict[key] = tensor

    if loaded_bytes != total_size:
        raise ValueError(
            f"Sharded checkpoint tensor byte total mismatch; expected {total_size}, loaded {loaded_bytes}."
        )
    return state_dict
