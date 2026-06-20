"""
processing_prismatic.py

HuggingFace-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. Default configuration
specifies `siglip-224px+7b`.
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === Image Processing ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


import torch.nn.functional as F

def _is_pil_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], Image.Image)

def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Accept CHW or BCHW (also tolerates HWC/BHWC by auto-transpose if channel dim==3). Return BCHW."""
    if x.ndim == 3:
        # CHW or HWC
        if x.shape[0] == 3:     # CHW
            x = x.unsqueeze(0)
        elif x.shape[-1] == 3:  # HWC
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Expected CHW or HWC with C=3; got {tuple(x.shape)}")
    elif x.ndim == 4:
        # BCHW or BHWC
        if x.shape[1] == 3:     # BCHW
            pass
        elif x.shape[-1] == 3:  # BHWC
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected BCHW or BHWC with C=3; got {tuple(x.shape)}")
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D")
    return x

def _ensure_float01(x: torch.Tensor) -> torch.Tensor:
    """Map uint8 [0,255] -> float32 [0,1]. Leave float types as-is but cast to float32."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif not x.is_floating_point():
        x = x.float()
    return x

def _letterbox_pad_tensor(img: torch.Tensor, fill_rgb01: Tuple[float, float, float]) -> torch.Tensor:
    """img: CHW tensor in [0,1]; pad to square with symmetric borders."""
    c, h, w = img.shape
    max_wh = max(h, w)
    pad_vert = (max_wh - h)
    pad_horz = (max_wh - w)
    # (left, top, right, bottom)
    padding = (pad_horz // 2, pad_vert // 2, pad_horz - pad_horz // 2, pad_vert - pad_vert // 2)
    # TVF.pad expects CHW or HWC for tensors; fill can be scalar or sequence len==C
    return TVF.pad(img, padding, fill=[fill_rgb01[0], fill_rgb01[1], fill_rgb01[2]], padding_mode="constant")


def letterbox_pad_pil(image: Image.Image, padding_fill_value_255: Tuple[int, int, int]) -> Image.Image:
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
    return TVF.pad(image, padding, fill=padding_fill_value_255, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: Any,
    ) -> None:
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill_255, self.tvf_letterbox_fill_01 = False, None, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,
                crop_mode="center",
                is_training=False,
            )

            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    # convert back to python floats for JSON-serializable config
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )

        # default: no letterbox fill; set strategy-specific options
        if self.image_resize_strategy == "resize-naive":
            for idx in range(len(self.input_sizes)):
                self.tvf_resize_params[idx]["size"] = (self.input_sizes[idx][-1], self.input_sizes[idx][-1])
        elif self.image_resize_strategy == "letterbox":
            self.tvf_do_letterbox = True
            # PIL path uses 0-255 ints; tensor path uses 0-1 floats
            # When fused backbone, each branch uses its own mean
            # Here we store for the first branch; padding is applied before stacking, so same fill is fine
            # or choose per-branch—both are acceptable (difference is negligible pre-normalize)
            fill0 = tuple(int(x * 255) for x in self.means[0])
            fill0f = tuple(float(x) for x in self.means[0])
            self.tvf_letterbox_fill_255 = fill0
            self.tvf_letterbox_fill_01 = fill0f
        elif self.image_resize_strategy == "resize-crop":
            pass
        else:
            raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        super().__init__(**kwargs)

    # -----------------------
    # PIL single image path
    # -----------------------
    def _apply_transform_pil(self, img: Image.Image) -> torch.Tensor:
        if self.tvf_do_letterbox:
            img = letterbox_pad_pil(img, self.tvf_letterbox_fill_255)

        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)  # [0,1] float32 CHW
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)
        return torch.vstack(imgs_t)  # [3 or 6, H, W]

    # -----------------------
    # Tensor single image path (CHW in [0,1] or uint8)
    # -----------------------
    def _apply_transform_tensor_chw(self, img_chw: torch.Tensor) -> torch.Tensor:
        img_chw = _ensure_float01(img_chw)
        if self.tvf_do_letterbox:
            img_chw = _letterbox_pad_tensor(img_chw, self.tvf_letterbox_fill_01)

        imgs_t = []
        for idx in range(len(self.input_sizes)):
            # TVF.resize/center_crop accept CHW tensors
            img_idx = TVF.resize(img_chw, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            # already float [0,1]
            mean = torch.tensor(self.tvf_normalize_params[idx]["mean"], dtype=img_idx.dtype, device=img_idx.device)
            std = torch.tensor(self.tvf_normalize_params[idx]["std"], dtype=img_idx.dtype, device=img_idx.device)
            img_idx = TVF.normalize(img_idx, mean=mean, std=std)
            imgs_t.append(img_idx)
        return torch.vstack(imgs_t)  # [3 or 6, H, W]

    # -----------------------
    # Public preprocess
    # -----------------------
    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: Any,
    ) -> BatchFeature:

        # Case A: PIL or list of PIL
        if isinstance(images, Image.Image) or _is_pil_list(images):
            pil_list = images if isinstance(images, list) else [images]
            pixel_values = torch.stack([self._apply_transform_pil(img.convert("RGB")) for img in pil_list])

        # Case B: Tensor (CHW or BCHW; also accepts HWC/BHWC)
        elif isinstance(images, torch.Tensor):
            x = _to_bchw(images)
            x = _ensure_float01(x)  # map uint8->float01 if needed
            batch, _, _, _ = x.shape
            outs = []
            for i in range(batch):
                outs.append(self._apply_transform_tensor_chw(x[i]))
            pixel_values = torch.stack(outs, dim=0)  # [B, 3 or 6, 224, 224]

        else:
            raise TypeError(
                "Unsupported `images` type. Expected PIL.Image, List[PIL.Image], or torch.Tensor "
                "(CHW/BCHW or HWC/BHWC)."
            )

        # BatchFeature: keep tensor if user asks for 'pt'
        if return_tensors in (TensorType.PYTORCH, "pt"):
            return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)
        else:
            return BatchFeature(data={"pixel_values": pixel_values.float().cpu().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image], torch.Tensor], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)



# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
