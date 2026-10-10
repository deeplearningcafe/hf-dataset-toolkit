import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageOps
import json

logger = logging.getLogger(__name__)


class BaseDatasetAdapter(ABC):
    """Abstract base adapter for normalizing HuggingFace streaming samples."""

    def __init__(
        self,
        name: str,
        min_resolution: int = 512,
        avif_quality: int = 80,
    ):
        self.name = name
        self.min_resolution = min_resolution
        self.avif_quality = avif_quality

    @abstractmethod
    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        pass

    @staticmethod
    def _to_avif_bytes(
        image_input: Any, quality: int = 80
    ) -> Optional[Tuple[bytes, int, int]]:
        """Converts PIL image or raw bytes into validated AVIF byte array."""
        try:
            if isinstance(image_input, Image.Image):
                img = image_input
            elif isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, dict) and "bytes" in image_input:
                img = Image.open(io.BytesIO(image_input["bytes"]))
            else:
                return None

            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            buffer = io.BytesIO()
            img.save(buffer, format="AVIF", quality=quality)
            return buffer.getvalue(), w, h
        except Exception as e:
            logger.warning(f"Error encoding image to AVIF: {e}. Skipping.")
            return None

    @staticmethod
    def _to_jpeg_bytes(image_input: Any) -> Optional[Tuple[bytes, int, int]]:
        return BaseDatasetAdapter._to_avif_bytes(image_input, quality=80)


class FluxReasonAdapter(BaseDatasetAdapter):
    """Adapter for LucasFang/FLUX-Reason-6M with category extraction."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        res = self._to_avif_bytes(raw_sample.get("image"), quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        prompt = (
            raw_sample.get("caption_detail")
            or raw_sample.get("caption_thought")
            or raw_sample.get("caption", "")
        )
        sample_id = str(raw_sample.get("id", f"flux_{raw_sample.get('index')}"))
        category = str(
            raw_sample.get("task")
            or raw_sample.get("category")
            or raw_sample.get("type", "general")
        )
        return img_bytes, str(prompt), sample_id, w, h, category


class ImageNet22kWdsAdapter(BaseDatasetAdapter):
    """Adapter for timm/imagenet-22k-wds with synset decoding."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = raw_sample.get("jpg") or raw_sample.get("png")
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        meta = raw_sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}

        cls_name = meta.get("class_name", "") if isinstance(meta, dict) else ""
        if not cls_name and "cls" in raw_sample:
            cls_name = str(raw_sample["cls"])

        sample_id = str(raw_sample.get("__key__", ""))
        return img_bytes, "", sample_id, w, h, str(cls_name)


class GPTEditAdapter(BaseDatasetAdapter):
    """Adapter for UCSC-VLAA/gpt-edit-simpler extracting base input."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        # if no input use output
        raw_img = raw_sample.get("input") or raw_sample.get("output")
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(raw_sample.get("id", ""))
        return img_bytes, "", sample_id, w, h, "general"


class TextAtlasAdapter(BaseDatasetAdapter):
    """Adapter for CSU-JPG/TextAtlas5M subsets."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        res = self._to_avif_bytes(raw_sample.get("image"), quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        prompt = raw_sample.get("annotation") or raw_sample.get("text", "")
        sample_id = str(raw_sample.get("global_id", f"ta5m_{hash(str(prompt))}"))
        return img_bytes, str(prompt), sample_id, w, h, "ocr"


class MidjourneyV6Adapter(BaseDatasetAdapter):
    """Adapter for Photoroom/midjourney-v6-recap."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        res = self._to_avif_bytes(raw_sample.get("image"), quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        prompt = raw_sample.get("prompt", "")
        sample_id = str(raw_sample.get("id", f"mj6_{hash(str(prompt))}"))
        return img_bytes, str(prompt), sample_id, w, h, "general"


class PexelsAdapter(BaseDatasetAdapter):
    """Adapter for animetimm/pexels-tagger-v0-w640-ws-full."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = (
            raw_sample.get("image") or raw_sample.get("jpg") or raw_sample.get("png")
        )
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(
            raw_sample.get("image_id")
            or raw_sample.get("id")
            or raw_sample.get("__key__", "")
        )
        return img_bytes, "", sample_id, w, h, "general"


class RenderedTextAdapter(BaseDatasetAdapter):
    """Adapter for wendlerc/RenderedText synthetic OCR benchmark."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = raw_sample.get("png") or raw_sample.get("image")
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(
            raw_sample.get("id")
            or raw_sample.get("image_id")
            or raw_sample.get("__key__", "")
        )
        return img_bytes, "", sample_id, w, h, "ocr"


class MegalithAdapter(BaseDatasetAdapter):
    """Adapter for Megalith images supporting cached and byte payloads."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = (
            raw_sample.get("image") or raw_sample.get("bytes") or raw_sample.get("jpg")
        )
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(
            raw_sample.get("id") or raw_sample.get("key") or raw_sample.get("url", "")
        )
        return img_bytes, "", sample_id, w, h, "general"
