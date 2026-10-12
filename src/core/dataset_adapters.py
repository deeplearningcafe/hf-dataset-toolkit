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
    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        """Extracts (raw_img_payload, prompt, sample_id, w, h, category)."""
        pass

    def encode_image(self, image_payload: Any) -> Optional[Tuple[bytes, int, int]]:
        """Encodes validated image payload into AVIF format."""
        return self._to_avif_bytes(image_payload, quality=self.avif_quality)

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        meta = self.extract_metadata(raw_sample)
        if not meta:
            return None
        raw_img, prompt, sample_id, w, h, category = meta
        res = self.encode_image(raw_img)
        if not res:
            return None
        img_bytes, final_w, final_h = res
        return img_bytes, prompt, sample_id, final_w, final_h, category

    @staticmethod
    def _extract_image_payload(
        raw_sample: Dict[str, Any],
        preferred_keys: Tuple[str, ...] = (
            "webp",
            "image",
            "jpg",
            "jpeg",
            "png",
            "bytes",
            "input",
            "output",
        ),
    ) -> Any:
        """Finds raw image payload across diverse WebDataset/Parquet keys."""
        for k in preferred_keys:
            val = raw_sample.get(k)
            if val is not None:
                return val
        return None

    @staticmethod
    def _probe_image_dims(image_input: Any) -> Optional[Tuple[int, int]]:
        """Extracts (width, height) without decoding pixel buffers."""
        try:
            if isinstance(image_input, Image.Image):
                return image_input.size
            if isinstance(image_input, (bytes, bytearray)):
                with Image.open(io.BytesIO(image_input)) as img:
                    return img.size
            if isinstance(image_input, dict) and "bytes" in image_input:
                with Image.open(io.BytesIO(image_input["bytes"])) as img:
                    return img.size
        except Exception:
            return None
        return None

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
        scores = {
            "composition": raw_sample.get("score_composition", 0),
            "entity": raw_sample.get("score_entity", 0),
            "text": raw_sample.get("score_text", 0),
            "imaginative": raw_sample.get("score_imaginative", 0),
            "style": raw_sample.get("score_style", 0),
            "abstract": raw_sample.get("score_abstract", 0),
            "original": raw_sample.get("score_original", 0),
        }
        category = str(max(scores, key=scores.get))
        return img_bytes, str(prompt), sample_id, w, h, category

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = raw_sample.get("image")
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        prompt = (
            raw_sample.get("caption_detail")
            or raw_sample.get("caption_thought")
            or raw_sample.get("caption", "")
        )
        sample_id = str(raw_sample.get("id", f"flux_{raw_sample.get('index')}"))
        scores = {
            "composition": raw_sample.get("score_composition", 0),
            "entity": raw_sample.get("score_entity", 0),
            "text": raw_sample.get("score_text", 0),
            "imaginative": raw_sample.get("score_imaginative", 0),
            "style": raw_sample.get("score_style", 0),
            "abstract": raw_sample.get("score_abstract", 0),
            "original": raw_sample.get("score_original", 0),
        }
        category = str(max(scores, key=scores.get))
        return raw_img, str(prompt), sample_id, w, h, category


class ImageNet22kWdsAdapter(BaseDatasetAdapter):
    """Adapter for timm/imagenet-22k-wds with fast metadata pruning."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        meta = raw_sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}

        w, h = 0, 0
        if isinstance(meta, dict):
            w = int(meta.get("width", 0))
            h = int(meta.get("height", 0))

        if w > 0 and h > 0 and min(w, h) < self.min_resolution:
            return None

        raw_img = self._extract_image_payload(
            raw_sample, ("jpg", "jpeg", "png", "webp", "image")
        )
        if raw_img is None:
            return None

        if w == 0 or h == 0:
            dims = self._probe_image_dims(raw_img)
            if not dims or min(dims[0], dims[1]) < self.min_resolution:
                return None
            w, h = dims

        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, final_w, final_h = res

        cls_name = meta.get("class_name", "") if isinstance(meta, dict) else ""
        if not cls_name and "cls" in raw_sample:
            cls_val = raw_sample["cls"]
            if isinstance(cls_val, (bytes, bytearray)):
                try:
                    cls_name = cls_val.decode("utf-8").strip()
                except Exception:
                    cls_name = str(cls_val)
            else:
                cls_name = str(cls_val)

        sample_id = str(raw_sample.get("__key__", ""))
        return img_bytes, "", sample_id, final_w, final_h, str(cls_name)

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        meta = raw_sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}

        w, h = 0, 0
        if isinstance(meta, dict):
            w = int(meta.get("width", 0))
            h = int(meta.get("height", 0))

        if w > 0 and h > 0 and min(w, h) < self.min_resolution:
            return None

        raw_img = self._extract_image_payload(
            raw_sample, ("jpg", "jpeg", "png", "webp", "image")
        )
        if raw_img is None:
            return None

        if w == 0 or h == 0:
            dims = self._probe_image_dims(raw_img)
            if not dims or min(dims[0], dims[1]) < self.min_resolution:
                return None
            w, h = dims

        cls_name = meta.get("class_name", "") if isinstance(meta, dict) else ""
        if not cls_name and "cls" in raw_sample:
            cls_val = raw_sample["cls"]
            if isinstance(cls_val, (bytes, bytearray)):
                try:
                    cls_name = cls_val.decode("utf-8").strip()
                except Exception:
                    cls_name = str(cls_val)
            else:
                cls_name = str(cls_val)

        sample_id = str(raw_sample.get("__key__", ""))
        return raw_img, "", sample_id, w, h, str(cls_name)


class GPTEditAdapter(BaseDatasetAdapter):
    """Adapter for UCSC-VLAA/gpt-edit-simpler extracting base input."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        # i1 captions use output
        raw_img = raw_sample.get("output")  # or raw_sample.get("input")
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(raw_sample.get("id", ""))
        return img_bytes, "", sample_id, w, h, "general"

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = raw_sample.get("output") or raw_sample.get("input")
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        sample_id = str(raw_sample.get("id", ""))
        return raw_img, "", sample_id, w, h, "general"


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

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = raw_sample.get("image")
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        prompt = raw_sample.get("annotation") or raw_sample.get("text", "")
        sample_id = str(raw_sample.get("global_id", f"ta5m_{hash(str(prompt))}"))
        return raw_img, str(prompt), sample_id, w, h, "ocr"


class MidjourneyV6Adapter(BaseDatasetAdapter):
    """Adapter for Photoroom/midjourney-v6-recap."""

    def __init__(
        self,
        name: str,
        min_resolution: int = 512,
        avif_quality: int = 80,
    ):
        super().__init__(name, min_resolution, avif_quality)
        self._last_id: Optional[str] = None
        self._sub_idx: int = 0

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

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = raw_sample.get("image")
        if raw_img is None:
            return None

        raw_id = raw_sample.get("id")
        if raw_id is None:
            return None

        raw_id_str = str(raw_id).strip()
        if raw_id_str == self._last_id:
            self._sub_idx += 1
        else:
            self._last_id = raw_id_str
            self._sub_idx = 0

        try:
            base_id = str(int(raw_id_str))
        except (ValueError, TypeError):
            base_id = raw_id_str.lstrip("0") or "0"

        sample_id = f"{base_id}_{self._sub_idx}"

        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        prompt = raw_sample.get("prompt", "")
        return raw_img, str(prompt), sample_id, w, h, "general"


class PexelsAdapter(BaseDatasetAdapter):
    """Adapter for animetimm/pexels-tagger-v0-w640-ws-full."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("webp", "image", "jpg", "jpeg", "png")
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

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("webp", "image", "jpg", "jpeg", "png")
        )
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        sample_id = str(
            raw_sample.get("image_id")
            or raw_sample.get("id")
            or raw_sample.get("__key__", "")
        )
        return raw_img, "", sample_id, w, h, "general"


class RenderedTextAdapter(BaseDatasetAdapter):
    """Adapter for wendlerc/RenderedText synthetic OCR benchmark."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("png", "webp", "jpg", "jpeg", "image")
        )
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

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("png", "webp", "jpg", "jpeg", "image")
        )
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        sample_id = str(
            raw_sample.get("id")
            or raw_sample.get("image_id")
            or raw_sample.get("__key__", "")
        )
        return raw_img, "", sample_id, w, h, "ocr"


class MegalithAdapter(BaseDatasetAdapter):
    """Adapter for Megalith images supporting cached and byte payloads."""

    def extract_sample(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[bytes, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("image", "bytes", "jpg", "jpeg", "webp", "png")
        )
        res = self._to_avif_bytes(raw_img, quality=self.avif_quality)
        if not res:
            return None
        img_bytes, w, h = res
        if min(w, h) < self.min_resolution:
            return None

        sample_id = str(
            raw_sample.get("id")
            or raw_sample.get("key")
            or raw_sample.get("__key__", "")
            or raw_sample.get("url", "")
        )
        return img_bytes, "", sample_id, w, h, "general"

    def extract_metadata(
        self, raw_sample: Dict[str, Any]
    ) -> Optional[Tuple[Any, str, str, int, int, str]]:
        raw_img = self._extract_image_payload(
            raw_sample, ("image", "bytes", "jpg", "jpeg", "webp", "png")
        )
        if raw_img is None:
            return None
        dims = self._probe_image_dims(raw_img)
        if not dims or min(dims[0], dims[1]) < self.min_resolution:
            return None
        w, h = dims

        sample_id = str(
            raw_sample.get("id")
            or raw_sample.get("key")
            or raw_sample.get("__key__", "")
            or raw_sample.get("url", "")
        )
        return raw_img, "", sample_id, w, h, "general"
