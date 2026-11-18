from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class OfficialAssetContent:
    path: str
    size: int
    git_oid: str
    lfs_sha256: str | None = None


@dataclass(frozen=True)
class OfficialAssetProvenance:
    asset_role: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    content: tuple[OfficialAssetContent, ...]


def _content(
    path: str,
    size: int,
    git_oid: str,
    lfs_sha256: str | None = None,
) -> OfficialAssetContent:
    return OfficialAssetContent(path, size, git_oid, lfs_sha256)


_AVATAR_REPO_ID = "meituan-longcat/LongCat-Video-Avatar-1.5"
_AVATAR_REVISION = "92016c71d5d318d0f5d84e4db30015a571484ab6"  # pragma: allowlist secret
_BASE_REPO_ID = "meituan-longcat/LongCat-Video"
_BASE_REVISION = "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"  # pragma: allowlist secret

AVATAR_SHARDED_PROVENANCE = OfficialAssetProvenance(
    asset_role="official_sharded",
    repo_id=_AVATAR_REPO_ID,
    revision=_AVATAR_REVISION,
    allow_patterns=(
        "base_model/config.json",
        "base_model/diffusion_pytorch_model.safetensors.index.json",
        "base_model/diffusion_pytorch_model-*.safetensors",
        "scheduler/*",
        "lora/dmd_lora.safetensors",
    ),
    content=(
        _content("base_model/config.json", 853, "666c43c40d8fe6dd61836fcacc807eb418c3235b"),  # pragma: allowlist secret
        _content(
            "base_model/diffusion_pytorch_model-00001-of-00006.safetensors",
            5038262208,
            "eccc662947f7a14dc6c829a1d745d0b3d22c965b",  # pragma: allowlist secret
            "c6140bdeb791620f2caa1f7c4d931dd4499bbafd148acdd65f1cdc254bd45eb2",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model-00002-of-00006.safetensors",
            5355725552,
            "64b45f37adbb6c4c06370fc360fdeb9e12296cb6",  # pragma: allowlist secret
            "90f71c9038c3d5fc596cc2bb2fc74c598b289bc76f57096277a42ff5bc9c37d9",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model-00003-of-00006.safetensors",
            5458518600,
            "be42452664bbb0fb5af3327a356275153f82661d",  # pragma: allowlist secret
            "bf69617758bfb0a2ca73d97341725dada77cc5fd2f07d07806beac3ecc5e0e6a",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model-00004-of-00006.safetensors",
            5265572392,
            "d74856cce0e338dac1f171df7ca7754b02a65024",  # pragma: allowlist secret
            "0540804fa47d2558ddc88d19dcf4a58ff7739673ec4da56cb2b1c5623810c57c",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model-00005-of-00006.safetensors",
            5353652768,
            "19d42c29a6249905d0928e160cbca96ba045dd7f",  # pragma: allowlist secret
            "91eedc26cf8a11a147cb0b29d640ccbd0c411c1b5f1949fad12003f48a9906c8",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model-00006-of-00006.safetensors",
            5234276864,
            "d90dda0dc3638ed2f745bcd30bd25be34181c4fd",  # pragma: allowlist secret
            "5fdb0fbff4bc75ea9bf1573c3551a974da4ab05b64b7cdd835a9b0d377374d1d",  # pragma: allowlist secret
        ),
        _content(
            "base_model/diffusion_pytorch_model.safetensors.index.json",
            153151,
            "22cf9e9603297b05b90ac9a17cdd2d00e985ecc4",  # pragma: allowlist secret
        ),
        _content(
            "lora/dmd_lora.safetensors",
            2523077984,
            "7858c1b776bf9a0f7a3ab3c445b1fae5e4919035",  # pragma: allowlist secret
            "d969115a7f3fbc212a277ed4399f77a1a065d170148c6d4bc272e980cd3e907a",  # pragma: allowlist secret
        ),
        _content("scheduler/scheduler_config.json", 213, "a11b9a003af3c9e300f1857df3acc660dafeec0f"),  # pragma: allowlist secret
    ),
)

AVATAR_INT8_SHARDED_PROVENANCE = OfficialAssetProvenance(
    asset_role="official_int8_sharded",
    repo_id=_AVATAR_REPO_ID,
    revision=_AVATAR_REVISION,
    allow_patterns=(
        "base_model_int8/config.json",
        "base_model_int8/quantization_config.json",
        "base_model_int8/quantized_model.safetensors.index.json",
        "base_model_int8/quantized_model-*.safetensors",
        "scheduler/*",
        "lora/dmd_lora.safetensors",
    ),
    content=(
        _content("base_model_int8/config.json", 853, "666c43c40d8fe6dd61836fcacc807eb418c3235b"),  # pragma: allowlist secret
        _content(
            "base_model_int8/quantization_config.json",
            193,
            "8367e2973ccc39bd6c871e16d2c7843a37b73f20",  # pragma: allowlist secret
        ),
        _content(
            "base_model_int8/quantized_model-00001-of-00004.safetensors",
            4264635792,
            "9f560b415b85adeb3366d1d25da2c4c1ca058b1b",  # pragma: allowlist secret
            "ccf575d8cdf8e762272e2d4e52ae1a7c0b5d1fa81e26dfa4592867de4dd9a4fd",  # pragma: allowlist secret
        ),
        _content(
            "base_model_int8/quantized_model-00002-of-00004.safetensors",
            4275232472,
            "19331fa128a1376de18c94ce0f9cde499a572037",  # pragma: allowlist secret
            "af6ddb737ad66d12fd5892adee568c14314143ebd4388d3ab9cc6065754b3688",  # pragma: allowlist secret
        ),
        _content(
            "base_model_int8/quantized_model-00003-of-00004.safetensors",
            4275232472,
            "8441856e46a07603efee9fc58baf14a86e4b1fa4",  # pragma: allowlist secret
            "6a349b76592b4752c2967235af457824d8938e3514bad8e78cfa86009e8a9bf5",  # pragma: allowlist secret
        ),
        _content(
            "base_model_int8/quantized_model-00004-of-00004.safetensors",
            3065282200,
            "15bf93a4defdee40859a9ed53ffebe5b801048fd",  # pragma: allowlist secret
            "ab54b648a3f6a53946f07dd0a21441e2a8cb1aa8da17996b5f3f7f2e1370705b",  # pragma: allowlist secret
        ),
        _content(
            "base_model_int8/quantized_model.safetensors.index.json",
            201884,
            "5fe9d29b77315bca6b75d57efb4b61a661c5b678",  # pragma: allowlist secret
        ),
        _content(
            "lora/dmd_lora.safetensors",
            2523077984,
            "7858c1b776bf9a0f7a3ab3c445b1fae5e4919035",  # pragma: allowlist secret
            "d969115a7f3fbc212a277ed4399f77a1a065d170148c6d4bc272e980cd3e907a",  # pragma: allowlist secret
        ),
        _content("scheduler/scheduler_config.json", 213, "a11b9a003af3c9e300f1857df3acc660dafeec0f"),  # pragma: allowlist secret
    ),
)

TEXT_ENCODER_PROVENANCE = OfficialAssetProvenance(
    asset_role="official_longcat_text_encoder",
    repo_id=_BASE_REPO_ID,
    revision=_BASE_REVISION,
    allow_patterns=("tokenizer/*", "text_encoder/*"),
    content=(
        _content("text_encoder/config.json", 854, "2fd01c57dddd8ae386d518c69c087c8ba8c73804"),  # pragma: allowlist secret
        _content(
            "text_encoder/model-00001-of-00005.safetensors",
            4972389712,
            "383f168e1318f0f8d6777b1ba5e88546c1273c46",  # pragma: allowlist secret
            "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99",  # pragma: allowlist secret
        ),
        _content(
            "text_encoder/model-00002-of-00005.safetensors",
            4899225672,
            "05f1cf9934e3ec7e228b51a93b56875cee468b86",  # pragma: allowlist secret
            "481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad",  # pragma: allowlist secret
        ),
        _content(
            "text_encoder/model-00003-of-00005.safetensors",
            4966309504,
            "3ea8c93410053a1d42cde624bdf8faf011d329e3",  # pragma: allowlist secret
            "f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3",  # pragma: allowlist secret
        ),
        _content(
            "text_encoder/model-00004-of-00005.safetensors",
            4999880704,
            "9c4d6361dc5ed76d8c129e9c7ad1c64ec7045637",  # pragma: allowlist secret
            "a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9",  # pragma: allowlist secret
        ),
        _content(
            "text_encoder/model-00005-of-00005.safetensors",
            2885866152,
            "75aa52c300dafa768721d112ac5a7f11512156ff",  # pragma: allowlist secret
            "7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea",  # pragma: allowlist secret
        ),
        _content(
            "text_encoder/model.safetensors.index.json",
            22476,
            "60ece61b46ecb3e6a5b705ea304bc97535317c2a",  # pragma: allowlist secret
        ),
        _content(
            "tokenizer/special_tokens_map.json",
            7079,
            "2ed25bf989a28d20b5d4b5822fbc24666d12a6f7",  # pragma: allowlist secret
        ),
        _content(
            "tokenizer/spiece.model",
            4548313,
            "2fe5f347e9f9367585589ae89e997dfbd5cf802c",  # pragma: allowlist secret
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",  # pragma: allowlist secret
        ),
        _content(
            "tokenizer/tokenizer.json",
            16837459,
            "9c6427ab2f801c942da4ee9740f1a750d3978a3b",  # pragma: allowlist secret
            "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef",  # pragma: allowlist secret
        ),
        _content(
            "tokenizer/tokenizer_config.json",
            61758,
            "09d434f9457238f697f4c208aab47f58caa15bfe",  # pragma: allowlist secret
        ),
    ),
)

OFFICIAL_ASSET_PROVENANCE: Mapping[str, OfficialAssetProvenance] = MappingProxyType(
    {
        AVATAR_SHARDED_PROVENANCE.asset_role: AVATAR_SHARDED_PROVENANCE,
        AVATAR_INT8_SHARDED_PROVENANCE.asset_role: AVATAR_INT8_SHARDED_PROVENANCE,
        TEXT_ENCODER_PROVENANCE.asset_role: TEXT_ENCODER_PROVENANCE,
    }
)
