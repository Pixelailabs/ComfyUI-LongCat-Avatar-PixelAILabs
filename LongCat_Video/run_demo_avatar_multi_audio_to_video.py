import math

import PIL.Image
import librosa
import numpy as np
import torch
from diffusers.utils import load_image

from .avatar_runtime_boundary import (
    append_continuation_frames,
    build_audio_center_indices,
    frames_to_output_tensor,
    generate_legacy_uid,
    move_separator_output,
    pipeline_output_to_frames,
    prepare_generation_settings,
    release_cuda_cache,
    slice_and_concatenate_audio_embeddings,
    trim_generated_output,
    validate_final_output_tensor,
)
from .debug_profile import ensure_debug_profiler
from .model_contract import AVATAR_V15
from .sampler_contract import expected_output_frames


def torch_gc():
    release_cuda_cache(torch)


def generate_random_uid():
    return generate_legacy_uid()


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    if source_path is None:
        return None

    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None

    return move_separator_output(outputs, target_path, audio_output_dir_temp)


def audio_prepare_multi(left_temp_vocal_path, right_temp_vocal_path, generate_duration, left_raw_speech_path, right_raw_speech_path, sample_rate=16000, audio_type='para'):
    left_speech_array, right_speech_array = None, None
    if left_temp_vocal_path is not None:
        left_speech_array, sr = librosa.load(left_temp_vocal_path, sr=sample_rate)
        left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate)

    if right_temp_vocal_path is not None:
        right_speech_array, sr = librosa.load(right_temp_vocal_path, sr=sample_rate)
        right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate)

    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        left_raw_speech_array = np.zeros_like(right_raw_speech_array)

    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        right_raw_speech_array = np.zeros_like(left_raw_speech_array)

    if audio_type == 'add':
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        merge_raw_speech = (
            np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)])
            + np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
        )
    elif audio_type == 'para':
        left_speech_array_ext = left_speech_array
        right_speech_array_ext = right_speech_array
        if len(left_speech_array_ext) != len(right_speech_array_ext):
            raise ValueError("audio_type='para' requires equal-length left and right audio.")
        merge_raw_speech = left_raw_speech_array + right_raw_speech_array
    else:
        raise NotImplementedError(f"Unsupported audio_type of {audio_type}")

    source_duration = len(left_speech_array_ext) / sr
    added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
    if added_sample_nums > 0:
        left_speech_array_ext = np.append(left_speech_array_ext, [0.0] * added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.0] * added_sample_nums)

    return left_speech_array_ext, right_speech_array_ext, merge_raw_speech


def resolve_person_bbox_coordinates(src_width, src_height, left_person_bbox, right_person_bbox):
    if left_person_bbox is None and right_person_bbox is None:
        face_scale = 0.1
        left_y_min, left_y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        half_width = src_width // 2
        left_x_min, left_x_max = int(half_width * face_scale), int(half_width * (1 - face_scale))
        right_x_min = int(half_width * face_scale + half_width)
        right_x_max = int(half_width * (1 - face_scale) + half_width)
    elif left_person_bbox is not None and right_person_bbox is not None:
        left_y_min, left_x_min, left_y_max, left_x_max = left_person_bbox
        right_y_min, right_x_min, right_y_max, right_x_max = right_person_bbox
    else:
        raise ValueError("Multi-audio generation requires both left and right person boxes, or neither.")
    return left_y_min, left_x_min, left_y_max, left_x_max, right_y_min, right_x_min, right_y_max, right_x_max


def generate_multi(pipe,condition,te_cond,device,seed,cond_image,resolution,
             text_guidance_scale,audio_guidance_scale,num_inference_steps,ref_img_index,mask_frame_range,use_distill,debug_profile=None):
    debug_profile = ensure_debug_profiler(debug_profile)
    audio_stride = condition['audio_stride']
    right_full_audio_emb = condition['full_audio_emb']
    left_full_audio_emb = condition['left_full_audio_emb']
    back_full_audio_emb = condition['back_full_audio_emb']
    num_segments = condition['num_segments']
    use_background_silent_audio = condition['use_background_silent_audio']
    left_person_bbox = condition['left_person_bbox']
    right_person_bbox = condition['right_person_bbox']
    other_person_bbox = condition['other_person_bbox']

    settings = prepare_generation_settings(
        pipe=pipe,
        device=device,
        seed=seed,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        ref_img_index=ref_img_index,
        mask_frame_range=mask_frame_range,
        use_distill=use_distill,
        fallback_model_type=AVATAR_V15,
        torch_module=torch,
    )
    num_frames = settings.num_frames
    num_cond_frames = settings.num_cond_frames
    num_inference_steps = settings.num_inference_steps
    text_guidance_scale = settings.text_guidance_scale
    audio_guidance_scale = settings.audio_guidance_scale
    ref_img_index = settings.ref_img_index
    mask_frame_range = settings.mask_frame_range
    generator = settings.generator

    audio_streams = [left_full_audio_emb, right_full_audio_emb]
    if use_background_silent_audio:
        audio_streams.append(back_full_audio_emb)
    with debug_profile.phase("segment_1.audio_slice", audio_stride=audio_stride):
        center_indices = build_audio_center_indices(
            full_audio_frames=left_full_audio_emb.shape[0],
            segment_idx=0,
            audio_stride=audio_stride,
        )
        audio_embs = slice_and_concatenate_audio_embeddings(
            audio_streams,
            center_indices,
            device=device,
            torch_module=torch,
        )

    image = load_image(cond_image)
    src_width, src_height = image.size

    background_mask = torch.zeros([src_height, src_width])
    human_mask1 = torch.zeros([src_height, src_width])
    human_mask2 = torch.zeros([src_height, src_width])
    (
        left_y_min,
        left_x_min,
        left_y_max,
        left_x_max,
        right_y_min,
        right_x_min,
        right_y_max,
        right_x_max,
    ) = resolve_person_bbox_coordinates(src_width, src_height, left_person_bbox, right_person_bbox)
    human_mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    human_mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background_mask += human_mask1
    background_mask += human_mask2
    background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
    total_mask = [human_mask1, human_mask2, background_mask]
    if use_background_silent_audio:
        for index in range(len(other_person_bbox) // 4):
            other_person_bbox_i = other_person_bbox[index * 4:(index + 1) * 4]
            other_person_mask = torch.zeros([src_height, src_width])
            other_person_mask[
                other_person_bbox_i[0]:other_person_bbox_i[2],
                other_person_bbox_i[1]:other_person_bbox_i[3],
            ] = 1
            total_mask.append(other_person_mask)
    ref_target_masks = torch.stack(total_mask, dim=0).to(device)

    output, latent = pipe.generate_ai2v(
        image=image,
        prompt=None,
        negative_prompt=None,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        output_type='both',
        generator=generator,
        audio_emb=audio_embs,
        ref_target_masks=ref_target_masks,
        use_distill=use_distill,
        prompt_embeds=te_cond["prompt_embeds"],
        negative_prompt_embeds=te_cond["negative_prompt_embeds"],
        text=te_cond["text"],
        debug_profile=debug_profile.child("segment_1.ai2v"),
    )
    with debug_profile.phase("segment_1.output_to_cpu_frames"):
        video = pipeline_output_to_frames(output, numpy_module=np, image_module=PIL.Image)
        output_tensor = frames_to_output_tensor(video, numpy_module=np, torch_module=torch)
    del output
    with debug_profile.phase("segment_1.torch_gc"):
        torch_gc()

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        print(f"Generating segment {segment_idx + 1}/{num_segments}...")
        segment_label = f"segment_{segment_idx + 1}"
        with debug_profile.phase(f"{segment_label}.audio_slice", audio_stride=audio_stride):
            center_indices = build_audio_center_indices(
                full_audio_frames=left_full_audio_emb.shape[0],
                segment_idx=segment_idx,
                audio_stride=audio_stride,
            )
            audio_embs = slice_and_concatenate_audio_embeddings(
                audio_streams,
                center_indices,
                device=device,
                torch_module=torch,
            )

        output, latent = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=getattr(pipe.dit, 'lora_mix_config', None) is None,
            # CRITICAL: official Avatar continuation keeps KV cache on GPU; CPU offload causes per-step host/device transfers and segment slowdown.
            offload_kv_cache=False,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_embs,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"],
            negative_prompt_embeds=te_cond["negative_prompt_embeds"],
            text=te_cond["text"],
            debug_profile=debug_profile.child(f"{segment_label}.avc"),
        )

        with debug_profile.phase(f"{segment_label}.output_to_cpu_frames"):
            new_video = pipeline_output_to_frames(output, numpy_module=np, image_module=PIL.Image)
            del output
            all_generated_frames, output_tensor = append_continuation_frames(
                all_generated_frames,
                new_video,
                num_cond_frames=num_cond_frames,
                numpy_module=np,
                torch_module=torch,
                expected_frames=expected_output_frames(segment_idx + 1),
            )
            current_video = new_video

    with debug_profile.phase("final_output_contract"):
        validate_final_output_tensor(output_tensor, num_segments=num_segments)
    # CRITICAL: generated segments include padded silent coverage; return only source-audio frames for lip-sync.
    with debug_profile.phase("trim_to_target_frames"):
        output_tensor = trim_generated_output(
            output_tensor,
            target_output_frames=condition.get("target_output_frames"),
        )
    return output_tensor
