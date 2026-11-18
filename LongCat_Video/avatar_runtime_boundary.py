from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

from .model_contract import normalize_sampling_parameters
from .sampler_contract import (
    NUM_COND_FRAMES,
    NUM_FRAMES,
    expected_output_frames,
    normalize_seed,
    resolve_resolution_dimensions,
    segment_audio_start,
    trim_output_tensor_to_target_frames,
    validate_continuation_parameters,
    validate_output_tensor_contract,
)


DEFAULT_AUDIO_CONTEXT_RADIUS = 2


@dataclass(frozen=True)
class AvatarGenerationSettings:
    height: int
    width: int
    seed: int
    num_inference_steps: int
    text_guidance_scale: float
    audio_guidance_scale: float
    ref_img_index: int
    mask_frame_range: int
    generator: Any
    num_frames: int = NUM_FRAMES
    num_cond_frames: int = NUM_COND_FRAMES


def prepare_generation_settings(
    *,
    pipe: Any,
    device: Any,
    seed: int,
    resolution: str,
    num_inference_steps: int,
    text_guidance_scale: float,
    audio_guidance_scale: float,
    ref_img_index: int,
    mask_frame_range: int,
    use_distill: bool,
    fallback_model_type: str,
    torch_module: Any,
) -> AvatarGenerationSettings:
    model_type = getattr(pipe, "model_type", fallback_model_type)
    effective_steps, effective_text_cfg, effective_audio_cfg = normalize_sampling_parameters(
        model_type,
        use_distill,
        num_inference_steps,
        text_guidance_scale,
        audio_guidance_scale,
    )
    height, width = resolve_resolution_dimensions(resolution)
    normalized_seed = normalize_seed(seed)
    normalized_ref_index, normalized_mask_range = validate_continuation_parameters(
        ref_img_index,
        mask_frame_range,
    )
    generator = torch_module.Generator(device=device)
    generator.manual_seed(normalized_seed)
    return AvatarGenerationSettings(
        height=height,
        width=width,
        seed=normalized_seed,
        num_inference_steps=effective_steps,
        text_guidance_scale=effective_text_cfg,
        audio_guidance_scale=effective_audio_cfg,
        ref_img_index=normalized_ref_index,
        mask_frame_range=normalized_mask_range,
        generator=generator,
    )


def build_audio_center_indices(
    *,
    full_audio_frames: int,
    segment_idx: int,
    audio_stride: int,
    context_radius: int = DEFAULT_AUDIO_CONTEXT_RADIUS,
) -> tuple[tuple[int, ...], ...]:
    full_audio_frames = int(full_audio_frames)
    context_radius = int(context_radius)
    if full_audio_frames < 1:
        raise ValueError("full_audio_frames must be at least 1.")
    if context_radius < 0:
        raise ValueError("context_radius must be non-negative.")
    audio_start_idx = segment_audio_start(segment_idx, audio_stride)
    centers = range(audio_start_idx, audio_start_idx + int(audio_stride) * NUM_FRAMES, int(audio_stride))
    offsets = range(-context_radius, context_radius + 1)
    return tuple(
        tuple(max(0, min(center + offset, full_audio_frames - 1)) for offset in offsets)
        for center in centers
    )


def slice_audio_embedding(
    embedding: Any,
    center_indices: Sequence[Sequence[int]],
    *,
    device: Any,
    torch_module: Any,
) -> Any:
    index_tensor = torch_module.tensor(center_indices)
    return embedding[index_tensor][None, ...].to(device)


def slice_and_concatenate_audio_embeddings(
    embeddings: Sequence[Any],
    center_indices: Sequence[Sequence[int]],
    *,
    device: Any,
    torch_module: Any,
) -> Any:
    if not embeddings:
        raise ValueError("Multi-audio slicing requires at least one embedding stream.")
    sliced = [
        slice_audio_embedding(
            embedding,
            center_indices,
            device=device,
            torch_module=torch_module,
        )
        for embedding in embeddings
    ]
    return torch_module.cat(sliced)


def pipeline_output_to_frames(
    output: Any,
    *,
    numpy_module: Any,
    image_module: Any,
) -> list[Any]:
    batch = output[0]
    arrays = [(batch[index] * 255).astype(numpy_module.uint8) for index in range(batch.shape[0])]
    return [image_module.fromarray(array) for array in arrays]


def frames_to_output_tensor(
    frames: Sequence[Any],
    *,
    numpy_module: Any,
    torch_module: Any,
    expected_frames: int | None = None,
    normalize_in_numpy: bool = False,
    validator: Callable[..., None] = validate_output_tensor_contract,
) -> Any:
    frame_array = numpy_module.array(frames)
    if normalize_in_numpy:
        frame_array = frame_array / 255.0
    output_tensor = torch_module.from_numpy(frame_array).float()
    if not normalize_in_numpy:
        output_tensor = output_tensor / 255.0
    if expected_frames is None:
        validator(output_tensor)
    else:
        validator(output_tensor, expected_frames=expected_frames)
    return output_tensor


def append_continuation_frames(
    generated_frames: Sequence[Any],
    continuation_frames: Sequence[Any],
    *,
    num_cond_frames: int,
    numpy_module: Any,
    torch_module: Any,
    expected_frames: int,
    validator: Callable[..., None] = validate_output_tensor_contract,
) -> tuple[list[Any], Any]:
    combined_frames = [*generated_frames, *continuation_frames[int(num_cond_frames) :]]
    output_tensor = frames_to_output_tensor(
        combined_frames,
        numpy_module=numpy_module,
        torch_module=torch_module,
        expected_frames=expected_frames,
        normalize_in_numpy=True,
        validator=validator,
    )
    return combined_frames, output_tensor


def finalize_output_tensor(
    output_tensor: Any,
    *,
    num_segments: int,
    target_output_frames: int | None,
    validator: Callable[..., None] = validate_output_tensor_contract,
    trimmer: Callable[[Any, int | None], Any] = trim_output_tensor_to_target_frames,
) -> Any:
    validate_final_output_tensor(output_tensor, num_segments=num_segments, validator=validator)
    return trim_generated_output(output_tensor, target_output_frames=target_output_frames, trimmer=trimmer)


def validate_final_output_tensor(
    output_tensor: Any,
    *,
    num_segments: int,
    validator: Callable[..., None] = validate_output_tensor_contract,
) -> None:
    validator(output_tensor, expected_frames=expected_output_frames(num_segments))


def trim_generated_output(
    output_tensor: Any,
    *,
    target_output_frames: int | None,
    trimmer: Callable[[Any, int | None], Any] = trim_output_tensor_to_target_frames,
) -> Any:
    return trimmer(output_tensor, target_output_frames)


def release_cuda_cache(torch_module: Any) -> None:
    torch_module.cuda.empty_cache()
    torch_module.cuda.ipc_collect()


def generate_legacy_uid(
    *,
    time_fn: Callable[[], float] = time.time,
    randint_fn: Callable[[int, int], int] = random.randint,
) -> str:
    timestamp_part = str(int(time_fn()))[-6:]
    random_part = str(randint_fn(100000, 999999))
    return timestamp_part + random_part


def move_separator_output(
    outputs: Sequence[str],
    target_path: str | Path,
    audio_output_dir_temp: str | Path,
    *,
    move: Callable[[str, str | Path], Any] = shutil.move,
) -> str | Path | None:
    if not outputs:
        return None
    source = (Path(audio_output_dir_temp) / "vocals" / outputs[0]).resolve().as_posix()
    move(source, target_path)
    return target_path
