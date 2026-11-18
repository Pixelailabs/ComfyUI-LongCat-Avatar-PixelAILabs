 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os

from comfy_api.latest import  io, ui
import folder_paths
from .node_utils import  clear_comfyui_cache,get_runtime_device,tensor2image,audio2path
from .LongCat_Video.run_demo_avatar_single_audio_to_video import load_longcat_video_model,generate,get_audio_vocal,get_audio_emb,load_audio_vocal
from .LongCat_Video.run_demo_avatar_multi_audio_to_video import generate_multi
from .LongCat_Video.audio_contract import (
    validate_audio_conditioning_payload,
    validate_longcat_avatar_whisper_model_name,
)
from .LongCat_Video.audio_crop import crop_audio_payload
from .LongCat_Video.bbox_contract import parse_person_boxes
from .LongCat_Video.model_contract import (
    OFFICIAL_V15_DISTILL_AUDIO_CFG,
    OFFICIAL_V15_DISTILL_STEPS,
    OFFICIAL_V15_DISTILL_TEXT_CFG,
)
from .LongCat_Video.performance_contract import (
    VAE_OFFLOAD_DEVICES,
    apply_runtime_plan,
    build_runtime_plan,
    cleanup_runtime_plan,
)
from .LongCat_Video.memory_lifecycle import attach_self_managed_lifecycle, open_runtime_lease
from .LongCat_Video.mux_audio_contract import materialized_mux_audio, select_mux_audio_source
from .LongCat_Video.sampler_contract import build_audio_window_payload, build_sampler_execution_request
from .LongCat_Video.text_conditioning import (
    DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
    TEXT_CONDITIONING_SOURCE_CLIP,
    TEXT_ENCODER_OFFLOAD_DEVICES,
    encode_official_text_conditioning,
    extract_scheduled_text_embedding,
    resolve_or_download_official_text_encoder_layout,
    validate_text_conditioning_payload,
)
from .LongCat_Video.video_output import normalize_mux_audio_path, save_muxed_video
from .LongCat_Video.attention_contract import ATTENTION_MODES
from .LongCat_Video.checkpoint_contract import (
    OFFICIAL_INT8_SHARDED,
    OFFICIAL_SHARDED,
    SINGLE_FILE_SAFETENSORS,
    build_download_manifest,
    download_missing_checkpoint_assets,
    inspect_checkpoint_source,
)
from .LongCat_Video.debug_profile import LongCatDebugProfiler

MAX_SEED = np.iinfo(np.int32).max
node_longcat_path = os.path.dirname(os.path.abspath(__file__))
weigths_longcat_current_path = os.path.join(folder_paths.models_dir, "longcat")
if not os.path.exists(weigths_longcat_current_path):
    os.makedirs(weigths_longcat_current_path)
folder_paths.add_model_folder_path("longcat", weigths_longcat_current_path) #  longcat dir

INFERENCE_WEIGHT_MODES = [
    SINGLE_FILE_SAFETENSORS,
    OFFICIAL_SHARDED,
    OFFICIAL_INT8_SHARDED,
]
DEFAULT_SINGLE_FILE_DIT = "LongCat-Video-Avatar-1.5-int8.safetensors"
MODEL_VERSIONS = ["1.6", "1.5"]
DEFAULT_OFFICIAL_CHECKPOINTS_BY_VERSION = {
    "1.6": {
        OFFICIAL_SHARDED: "LongCat-Video-Avatar-1.6/base_model/diffusion_pytorch_model.safetensors.index.json",
        OFFICIAL_INT8_SHARDED: "LongCat-Video-Avatar-1.6/base_model_int8/quantized_model.safetensors.index.json",
    },
    "1.5": {
        OFFICIAL_SHARDED: "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json",
        OFFICIAL_INT8_SHARDED: "LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json",
    },
}
DEFAULT_OFFICIAL_CHECKPOINTS = DEFAULT_OFFICIAL_CHECKPOINTS_BY_VERSION["1.6"]

# PixelAI Labs: fresh-install helpers (auto download of only the files the nodes need).
V16_REPO = "meituan-longcat/LongCat-Video-Avatar-1.6"
V16_DIR_NAME = "LongCat-Video-Avatar-1.6"
BASE_REPO = "meituan-longcat/LongCat-Video"
BASE_DIR_NAME = "LongCat-Video"
AUTO_FILE = "auto (models/longcat, downloads if missing)"


def _hf_snapshot(repo_id, local_dir, allow_patterns):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is needed for the automatic download: pip install huggingface_hub") from exc
    print(f"[PixelAI Labs] LongCat: downloading {allow_patterns} from {repo_id} to {local_dir}")
    snapshot_download(repo_id, local_dir=local_dir, allow_patterns=list(allow_patterns))


def _ensure_v16_weights(inference_weight_mode):
    root = os.path.join(weigths_longcat_current_path, V16_DIR_NAME)
    sub = "base_model_int8" if inference_weight_mode == OFFICIAL_INT8_SHARDED else "base_model"
    patterns = [f"{sub}/*", "loraA/*", "loraB/*", "config.json", "model_index.json", "scheduler/*", "base_model/config.json"]
    need = [os.path.join(root, sub), os.path.join(root, "loraA", "lora.safetensors"), os.path.join(root, "loraB", "lora.safetensors")]
    if not all(os.path.exists(x) for x in need):
        _hf_snapshot(V16_REPO, root, patterns)
    return root


def _auto_vae_path(auto_download):
    path = os.path.join(weigths_longcat_current_path, BASE_DIR_NAME, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.isfile(path):
        if not auto_download:
            raise FileNotFoundError(f"LongCat VAE not found: {path} (turn on auto_download_missing_weights)")
        _hf_snapshot(BASE_REPO, os.path.join(weigths_longcat_current_path, BASE_DIR_NAME), ["vae/*"])
    return path


def _auto_whisper_path():
    path = os.path.join(weigths_longcat_current_path, V16_DIR_NAME, "whisper-large-v3", "model.safetensors")
    if not os.path.isfile(path):
        _hf_snapshot(V16_REPO, os.path.join(weigths_longcat_current_path, V16_DIR_NAME),
                     ["whisper-large-v3/model.safetensors", "whisper-large-v3/*.json", "whisper-large-v3/*.txt"])
    return path


def _resolve_auto_attention(attention_mode):
    """'auto' picks the fastest backend that is installed: SageAttention, then FlashAttention 2, then PyTorch SDPA."""
    if attention_mode != "auto":
        return attention_mode
    from .LongCat_Video.attention_contract import inspect_attention_backend
    for mode in ("sageattn", "flash_attn_2"):
        try:
            if inspect_attention_backend(mode).available:
                return mode
        except Exception:
            pass
    return "sdpa"


def _resolve_single_file_model_name(inference_weight_mode):
    if inference_weight_mode == SINGLE_FILE_SAFETENSORS:
        return DEFAULT_SINGLE_FILE_DIT
    if inference_weight_mode in DEFAULT_OFFICIAL_CHECKPOINTS:
        return None
    raise ValueError(f"Unsupported inference_weight_mode: {inference_weight_mode}")


def _resolve_official_checkpoint_name(inference_weight_mode, model_version="1.6"):
    if inference_weight_mode == SINGLE_FILE_SAFETENSORS:
        return "none"
    try:
        return DEFAULT_OFFICIAL_CHECKPOINTS_BY_VERSION[model_version][inference_weight_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported inference_weight_mode: {inference_weight_mode}") from exc


def _resolve_official_checkpoint_path(
    inference_weight_mode,
    auto_download_missing_weights,
    model_version="1.6",
):
    official_checkpoint = _resolve_official_checkpoint_name(inference_weight_mode, model_version)
    selected_path = (
        folder_paths.get_full_path("longcat", official_checkpoint)
        if official_checkpoint != "none"
        else None
    )
    if selected_path is not None:
        inspection = inspect_checkpoint_source(inference_weight_mode, selected_path)
        if inspection.is_complete:
            return selected_path
        if not auto_download_missing_weights:
            raise FileNotFoundError(
                "Official checkpoint is incomplete; missing: "
                + ", ".join(inspection.missing_files)
            )

    if not auto_download_missing_weights:
        raise FileNotFoundError(
            "No complete official checkpoint was selected. Select an official "
            "checkpoint index under models/longcat or enable auto_download_missing_weights."
        )

    if model_version != "1.5":
        raise FileNotFoundError(
            "Avatar 1.6 weights not found. Put meituan-longcat/LongCat-Video-Avatar-1.6 "
            "(base_model_int8, loraA, loraB) in ComfyUI/models/longcat/LongCat-Video-Avatar-1.6/."
        )
    manifest = build_download_manifest(inference_weight_mode, weigths_longcat_current_path)
    download_missing_checkpoint_assets(manifest)
    inspection = inspect_checkpoint_source(inference_weight_mode, manifest.local_dir)
    if not inspection.is_complete:
        raise FileNotFoundError(
            "Official checkpoint download did not produce a complete checkpoint; missing: "
            + ", ".join(inspection.missing_files)
        )
    return manifest.local_dir


class LongCat_Video_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_Model",
            display_name="PixelAI Labs LongCat Avatar Model Loader",
            category="PixelAI Labs/LongCat Avatar",
            description="Loads LongCat-Video-Avatar 1.6 (or 1.5). 1.6 runs 8 steps: DMD LoRA (loraA) for the first mix_steps, OPSD LoRA (loraB) after.",
            inputs=[
                io.Combo.Input("model_version", options=MODEL_VERSIONS, default="1.6"),
                io.Combo.Input("inference_weight_mode",options=INFERENCE_WEIGHT_MODES, default=OFFICIAL_INT8_SHARDED),
                io.Combo.Input("attention_mode",options=ATTENTION_MODES),
                io.Boolean.Input("auto_download_missing_weights", default=True, tooltip="Download only the files this node needs into models/longcat (1.6: INT8 model 15.9 GB + two LoRAs 5 GB; VAE 0.5 GB)."),
                io.Combo.Input("vae",options= [AUTO_FILE] + folder_paths.get_filename_list("vae"), default=AUTO_FILE, tooltip="auto = the diffusers-format Wan VAE from meituan-longcat/LongCat-Video (models/longcat/LongCat-Video/vae). The usual ComfyUI wan_2.1_vae.safetensors does NOT work here."),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras"), tooltip="1.6: leave none to use the official loraB (OPSD). 1.5: the distill LoRA."),
                io.Int.Input("mix_steps", default=4, min=0, max=8, tooltip="1.6 only. First N of the 8 steps use the DMD LoRA mix; 0 = OPSD LoRA only (static)."),
                io.Float.Input("mix_alpha", default=1.0, min=0.0, max=1.0, step=0.05, tooltip="1.6 only. DMD weight during the mix steps (1.0 = DMD only)."),
                io.Boolean.Input("lora_on_gpu", default=False, optional=True, tooltip="Experimental: keep the speed LoRAs on the GPU while sampling. Ran out of memory in our 480p test on a 32 GB card, so leave it off."),
                io.Boolean.Input("compile_dit", default=False, optional=True, tooltip="torch.compile the transformer blocks (official 1.6 speed option). The first run with a new size is slower while it compiles; later runs are faster."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, model_version, inference_weight_mode,attention_mode,auto_download_missing_weights,vae,lora,mix_steps=4,mix_alpha=1.0,lora_on_gpu=False,compile_dit=False) -> io.NodeOutput:
        diffusion_models = _resolve_single_file_model_name(inference_weight_mode)
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models is not None else None
        attention_mode = _resolve_auto_attention(attention_mode)
        print(f"[PixelAI Labs] LongCat attention: {attention_mode}")
        vae_path = _auto_vae_path(bool(auto_download_missing_weights)) if vae in (AUTO_FILE, "none") else folder_paths.get_full_path("vae", vae)
        lora_path=folder_paths.get_full_path("loras",lora) if lora != "none" else None
        v16_root = os.path.join(weigths_longcat_current_path, V16_DIR_NAME)
        if model_version == "1.6" and inference_weight_mode in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED) and auto_download_missing_weights:
            _ensure_v16_weights(inference_weight_mode)
        if model_version == "1.6" and lora_path is None:
            lora_path = os.path.join(v16_root, "loraB", "lora.safetensors")
        if inference_weight_mode in (OFFICIAL_SHARDED, OFFICIAL_INT8_SHARDED):
            official_checkpoint_path = _resolve_official_checkpoint_path(
                inference_weight_mode,
                bool(auto_download_missing_weights),
                model_version,
            )
            model=load_longcat_video_model(
                None,
                vae_path,
                lora_path,
                node_longcat_path,
                use_int8=inference_weight_mode == OFFICIAL_INT8_SHARDED,
                checkpoint_source=inference_weight_mode,
                official_checkpoint_path=official_checkpoint_path,
                attention_mode=attention_mode,
            )
        else:
            model_path=dit_path
            selected_model_name = os.path.basename(model_path).lower() if model_path else ""
            use_int8 = selected_model_name.endswith(".safetensors") and "int8" in selected_model_name
            model=load_longcat_video_model(
                model_path,
                vae_path,
                lora_path,
                node_longcat_path,
                use_int8=use_int8,
                checkpoint_source=inference_weight_mode,
                attention_mode=attention_mode,
            )
        model.pixelai_model_version = model_version
        model.use_loudness_norm = model_version != "1.6"
        if model_version == "1.6" and int(mix_steps) > 0:
            # Loader already attached loraB (OPSD) under key "dmd"; add loraA (DMD) and switch per step.
            dmd_path = os.path.join(v16_root, "loraA", "lora.safetensors")
            if not os.path.isfile(dmd_path):
                raise FileNotFoundError(f"Avatar 1.6 DMD LoRA not found: {dmd_path}")
            print(f"[PixelAI Labs] LongCat 1.6 LoRA mix: DMD={dmd_path} first {mix_steps} steps (alpha {mix_alpha}), OPSD={lora_path}")
            model.dit.load_lora(dmd_path, "v16_dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            model.dit.enable_lora_mix("v16_dmd", "dmd", float(mix_alpha), int(mix_steps))
        model.dit.pixelai_lora_on_gpu = bool(lora_on_gpu)
        if compile_dit:
            # Same settings as the official 1.6 --compile_dit path.
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.accumulated_cache_size_limit = 1024
            print("[PixelAI Labs] LongCat: compiling DiT transformer blocks (first run is slower)")
            for block_idx, block in enumerate(model.dit.blocks):
                model.dit.blocks[block_idx] = torch.compile(
                    block,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False, "epilogue_fusion": False, "emulate_precision_casts": True},
                )
        attach_self_managed_lifecycle(model)
        return io.NodeOutput(model)


class LongCat_Video_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_Sampler",
            display_name="PixelAI Labs LongCat Avatar Sampler",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("te_cond"),
                io.Conditioning.Input("au_cond"),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional for at2v; required when stage_1 is ai2v.",
                ),
                io.Combo.Input("stage_1",options= ['ai2v', 'at2v']),
                io.Combo.Input("resolution",options= ['480p', '720p']),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input(
                    "steps",
                    display_name="steps (fixed 8, distill)",
                    tooltip="Avatar 1.5 DMD/distill always uses 8 inference steps.",
                    default=OFFICIAL_V15_DISTILL_STEPS,
                    min=OFFICIAL_V15_DISTILL_STEPS,
                    max=OFFICIAL_V15_DISTILL_STEPS,
                    step=1,
                    extra_dict={"disabled": True},
                ),
                io.Float.Input(
                    "text_guidance_scale",
                    display_name="text CFG (fixed, distill)",
                    tooltip="Avatar 1.5 DMD/distill always uses text CFG 1.0.",
                    default=OFFICIAL_V15_DISTILL_TEXT_CFG,
                    min=OFFICIAL_V15_DISTILL_TEXT_CFG,
                    max=OFFICIAL_V15_DISTILL_TEXT_CFG,
                    step=0.1,
                    extra_dict={"disabled": True},
                ),
                io.Float.Input(
                    "audio_guidance_scale",
                    display_name="audio CFG (fixed, distill)",
                    tooltip="Avatar 1.5 DMD/distill always uses audio CFG 1.0.",
                    default=OFFICIAL_V15_DISTILL_AUDIO_CFG,
                    min=OFFICIAL_V15_DISTILL_AUDIO_CFG,
                    max=OFFICIAL_V15_DISTILL_AUDIO_CFG,
                    step=0.1,
                    extra_dict={"disabled": True},
                ),
                io.Int.Input("ref_img_index", default=10, min=0, max=1024, step=1),
                io.Int.Input("mask_frame_range", default=3, min=0, max=1024, step=1),
                io.Int.Input("block_num", default=1, min=0, max=64,step=1),
                io.String.Input(
                    "mux_audio_path",
                    display_name="mux audio path (legacy)",
                    default="",
                    multiline=False,
                    tooltip="Deprecated migration input; use mux_audio. Paths are limited to ComfyUI input, output, or temp.",
                ),
                io.Combo.Input("offload_device", options=list(VAE_OFFLOAD_DEVICES), default="cpu"),
                io.Boolean.Input("debug_mode", default=False),
                io.Audio.Input(
                    "mux_audio",
                    optional=True,
                    tooltip="Optional native ComfyUI AUDIO to mux into the generated video.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="video_path"),
            ],
        )

    @classmethod
    def validate_inputs(cls, stage_1, input_types, mux_audio_path=""):
        if stage_1 not in ("ai2v", "at2v"):
            return "LongCat Avatar stage_1 must be either 'ai2v' or 'at2v'."
        if stage_1 == "ai2v" and "image" not in input_types:
            return "LongCat Avatar stage_1='ai2v' requires a connected image input."
        if normalize_mux_audio_path(mux_audio_path) and "mux_audio" in input_types:
            return "Connect either native mux_audio or legacy mux_audio_path, not both."
        return True

    @classmethod
    def execute(
        cls,
        model,
        te_cond,
        au_cond,
        image=None,
        stage_1="ai2v",
        resolution="480p",
        seed=0,
        steps=OFFICIAL_V15_DISTILL_STEPS,
        text_guidance_scale=OFFICIAL_V15_DISTILL_TEXT_CFG,
        audio_guidance_scale=OFFICIAL_V15_DISTILL_AUDIO_CFG,
        ref_img_index=10,
        mask_frame_range=3,
        block_num=1,
        mux_audio_path="",
        offload_device="cpu",
        debug_mode=False,
        mux_audio=None,
    ) -> io.NodeOutput:
        runtime_device = get_runtime_device()
        debug_profile = LongCatDebugProfiler(bool(debug_mode), label="sampler", device=runtime_device)
        with debug_profile.phase("validate_inputs"):
            request = build_sampler_execution_request(
                au_cond,
                image=image,
                stage_1=stage_1,
                resolution=resolution,
                seed=seed,
                steps=steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
                block_num=block_num,
                mux_audio_path=mux_audio_path,
                offload_device=offload_device,
            )
            mux_source = select_mux_audio_source(
                mux_audio,
                request.mux_audio_path,
                allowed_roots=(
                    folder_paths.get_input_directory(),
                    folder_paths.get_output_directory(),
                    folder_paths.get_temp_directory(),
                ),
            )
            validate_text_conditioning_payload(te_cond, require_negative=True)
            validate_audio_conditioning_payload(au_cond)
        with debug_profile.phase("build_runtime_plan"):
            runtime_plan = build_runtime_plan(runtime_device, request.block_num, request.offload_device)
            print(
                "[INFO] LongCat runtime plan: "
                f"block_num={getattr(runtime_plan, 'block_num', request.block_num)}, "
                f"streaming_prefetch_count={getattr(runtime_plan, 'streaming_prefetch_count', 'unknown')}, "
                f"move_dit_to_device={getattr(runtime_plan, 'move_dit_to_device', 'unknown')}, "
                f"offload_dit_after_generate={getattr(runtime_plan, 'offload_dit_after_generate', 'unknown')}, "
                f"vae_offload_device={getattr(runtime_plan, 'vae_offload_device', request.offload_device)}"
            )
        def apply_profiled_runtime(active_model, active_plan):
            with debug_profile.phase("apply_runtime_plan"):
                apply_runtime_plan(active_model, active_plan)

        def cleanup_profiled_runtime(active_model, active_plan):
            with debug_profile.phase("cleanup_runtime_plan"):
                cleanup_runtime_plan(active_model, active_plan)

        runtime_lease = open_runtime_lease(
            model,
            runtime_plan,
            apply_runtime=apply_profiled_runtime,
            cleanup_runtime=cleanup_profiled_runtime,
        )
        with runtime_lease:
            with debug_profile.phase("prepare_image"):
                cond_image = tensor2image(image) if request.stage_1 == "ai2v" else None
            with debug_profile.phase("generate", mode=request.mode):
                if request.mode == "multi":
                    image=generate_multi(model,au_cond,te_cond,runtime_device,request.seed,cond_image,request.resolution,
                        request.text_guidance_scale,request.audio_guidance_scale,request.steps,request.ref_img_index,request.mask_frame_range,model.use_distill,
                        debug_profile=debug_profile.child("multi"))
                else:
                    image=generate(model,au_cond,te_cond,runtime_device,request.seed,request.stage_1,cond_image,request.resolution,
                        request.text_guidance_scale,request.audio_guidance_scale,request.steps,request.ref_img_index,request.mask_frame_range,
                        model.use_distill,debug_profile=debug_profile.child("single"))
        with debug_profile.phase("save_muxed_video", enabled=mux_source.enabled, mode=request.mode):
            with materialized_mux_audio(
                mux_source,
                temp_dir=folder_paths.get_temp_directory(),
            ) as mux_audio_file:
                try:
                    video_path = save_muxed_video(
                        image,
                        enabled=mux_source.enabled,
                        output_dir=folder_paths.get_output_directory(),
                        audio_path=mux_audio_file,
                        prefix=None,
                        mode=request.mode,
                    )
                except Exception:
                    # CRITICAL: native temp paths must not escape through codec/subprocess exceptions.
                    if mux_source.kind == "native":
                        raise RuntimeError("Native mux audio video mux failed.") from None
                    raise
        return io.NodeOutput(image, video_path)

class LongCat_Video_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_TextEncode",
            display_name="PixelAI Labs LongCat Avatar Text Encode",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Clip.Input("clip", optional=True),
                io.String.Input("text_encoder_root", default=DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT, multiline=False),
                io.Boolean.Input("auto_download_missing_text_encoder", default=True),
                io.Combo.Input("offload_device", options=list(TEXT_ENCODER_OFFLOAD_DEVICES), default="cpu"),
                io.String.Input("prompt",default="A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.",multiline=True),
                io.String.Input("negative_prompt",default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards.",multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="te_cond"),
                ],
        )
    @classmethod
    def execute(
        cls,
        clip=None,
        text_encoder_root=DEFAULT_OFFICIAL_TEXT_ENCODER_ROOT,
        auto_download_missing_text_encoder=True,
        offload_device="cpu",
        prompt="",
        negative_prompt="",
    ) -> io.NodeOutput:
        runtime_device = get_runtime_device()
        if clip is None:
            # CRITICAL: official path is preferred; connected legacy CLIP input deliberately selects single-file UMT5 fallback.
            layout = resolve_or_download_official_text_encoder_layout(
                text_encoder_root,
                weigths_longcat_current_path,
                auto_download_missing_text_encoder=bool(auto_download_missing_text_encoder),
            )
            te_cond = encode_official_text_conditioning(
                layout=layout,
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=runtime_device,
                offload_device=offload_device,
                dtype=torch.bfloat16,
            )
            clear_comfyui_cache()
            return io.NodeOutput(te_cond)

        tokens = clip.tokenize(prompt)
        prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        prompt_embeds=extract_scheduled_text_embedding(prompt_embeds, "prompt_embeds")
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1, 1, seq_len, -1)
        tokens = clip.tokenize(negative_prompt)
        negative_prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        negative_prompt_embeds=extract_scheduled_text_embedding(negative_prompt_embeds, "negative_prompt_embeds")
        _, seq_len, _ = negative_prompt_embeds.shape
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1, 1, seq_len, -1)
        te_cond={
            "prompt_embeds":prompt_embeds.to(runtime_device,torch.bfloat16),
            "negative_prompt_embeds":negative_prompt_embeds.to(runtime_device,torch.bfloat16),
            "text":[prompt,negative_prompt],
            "conditioning_source": TEXT_CONDITIONING_SOURCE_CLIP,
        } # #torch.Size([1, 1, 512, 4096])
        validate_text_conditioning_payload(te_cond, require_negative=True)
        clear_comfyui_cache()

        return io.NodeOutput(te_cond)

class LongCat_Video_SM_Audio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_AudioEncode",
            display_name="PixelAI Labs LongCat Avatar Audio Encode",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
                io.Int.Input("save_fps", default=25, min=8, max=1024, step=1),
                io.Combo.Input("audio_type",options= ['para', 'add']),
                io.String.Input("p_box",default="",multiline=False),
                io.Combo.Input("model_version", options=MODEL_VERSIONS, default="1.6", tooltip="1.6 skips audio loudness normalisation, like the official 1.6 script."),
                io.Audio.Input("left_audio",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="au_cond"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,save_fps,audio_type,p_box,left_audio=None,num_segments=None,model_version="1.6") -> io.NodeOutput:
        parsed_p_box = parse_person_boxes(p_box)
        runtime_device = get_runtime_device()

        # IMPORTANT: legacy exported workflows may still carry num_segments; duration is now audio-driven.
        au_cond=get_audio_emb(audio_encoder,audio,left_audio,audio_type,save_fps,runtime_device,p_box=parsed_p_box,use_loudness_norm=(model_version != "1.6"))
        clear_comfyui_cache()
        return io.NodeOutput(au_cond)


class LongCat_Video_SM_AudioWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_AudioWindow",
            display_name="PixelAI Labs LongCat Avatar Audio Window",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Conditioning.Input("au_cond"),
                io.Int.Input("frames_processed", default=0, min=0, max=100000, step=1),
                io.Int.Input("num_frames", default=93, min=1, max=256, step=1),
                io.Int.Input("overlap", default=13, min=0, max=32, step=1),
                io.Combo.Input("if_not_enough_audio", options=["clamp", "mirror_from_end"]),
                io.Int.Input("ref_img_index", default=10, min=0, max=1024, step=1),
                io.Int.Input("mask_frame_range", default=3, min=0, max=1024, step=1),
            ],
            outputs=[
                io.Conditioning.Output(display_name="au_cond_window"),
            ],
        )

    @classmethod
    def execute(
        cls,
        au_cond,
        frames_processed,
        num_frames,
        overlap,
        if_not_enough_audio,
        ref_img_index,
        mask_frame_range,
    ) -> io.NodeOutput:
        au_cond_window = build_audio_window_payload(
            au_cond,
            frames_processed=frames_processed,
            num_frames=num_frames,
            overlap=overlap,
            if_not_enough_audio=if_not_enough_audio,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
        )
        clear_comfyui_cache()
        return io.NodeOutput(au_cond_window)


class LongCat_Video_SM_AudioCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_AudioCrop",
            display_name="PixelAI Labs LongCat Avatar Audio Crop",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("start_time", default="0:00", multiline=False),
                io.String.Input("end_time", default="1:00", multiline=False),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            has_intermediate_output=True,
        )

    @classmethod
    def execute(cls, audio, start_time="0:00", end_time="1:00") -> io.NodeOutput:
        cropped_audio = crop_audio_payload(audio, start_time=start_time, end_time=end_time)
        # IMPORTANT: keep preview generation on ComfyUI's native temp-audio path; downstream nodes receive AUDIO unchanged.
        return io.NodeOutput(cropped_audio, ui=ui.PreviewAudio(cropped_audio, cls=cls))

class LongCat_Video_SM_Vocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_VocalExtract",
            display_name="PixelAI Labs LongCat Avatar Vocal Extract",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="audio_path"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,) -> io.NodeOutput:
        audio_path,audio=get_audio_vocal(audio_encoder,audio2path(audio),folder_paths.get_output_directory())
        return io.NodeOutput(audio,audio_path)

class LongCat_Video_SM_WhisperModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_Whisper",
            display_name="PixelAI Labs LongCat Avatar Whisper Loader",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Combo.Input(
                    "audio_encoder",options=[AUTO_FILE] + folder_paths.get_filename_list("audio_encoders"), default=AUTO_FILE,
                    tooltip="auto = whisper-large-v3 from the LongCat-Video-Avatar-1.6 repo (3.1 GB, downloaded once). Or pick whisper-large-v3.safetensors from models/audio_encoders.",
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder) -> io.NodeOutput:
        # IMPORTANT: Avatar 1.5 is Whisper-only; Wav2Vec2 can load permissively and break lip sync.
        if audio_encoder == AUTO_FILE:
            a_checkpoint_path = _auto_whisper_path()
        else:
            validate_longcat_avatar_whisper_model_name(audio_encoder)
            a_checkpoint_path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder)
        from .LongCat_Video.longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        audio_encoder = get_audio_encoder(a_checkpoint_path, 'avatar-v1.5',os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"))
        audio_feature_extractor = get_audio_feature_extractor(os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"), 'avatar-v1.5')
        audio_encoder={"audio_encoder":audio_encoder,"audio_feature_extractor":audio_feature_extractor}
        return io.NodeOutput(audio_encoder)

class LongCat_Video_SM_VocalModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelAILabs_LongCatAvatar_VocalModel",
            display_name="PixelAI Labs LongCat Avatar Vocal Model",
            category="PixelAI Labs/LongCat Avatar",
            inputs=[
                io.Combo.Input(
                    "audio_encoder_vocal",options=["none"]+[i for i in folder_paths.get_filename_list("longcat") if i.endswith(".onnx")],
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder_vocal) -> io.NodeOutput:
        vocal_separator_path=folder_paths.get_full_path_or_raise("longcat", audio_encoder_vocal) if audio_encoder_vocal!="none" else None
        audio_encoder=load_audio_vocal(vocal_separator_path,folder_paths.get_output_directory(),weigths_longcat_current_path)
        return io.NodeOutput(audio_encoder)
