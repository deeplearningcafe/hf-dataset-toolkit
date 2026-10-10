from collections import Counter, deque
import gc
import logging
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
import yaml
import sys

from .core.bucketing import OnlineBucketManager
from .core.caption_loader import StreamingI1CaptionMap, load_i1_caption_map
from .core.dataset_adapters import (
    BaseDatasetAdapter,
    FluxReasonAdapter,
    GPTEditAdapter,
    ImageNet22kWdsAdapter,
    MegalithAdapter,
    MidjourneyV6Adapter,
    PexelsAdapter,
    RenderedTextAdapter,
    TextAtlasAdapter,
)
from .core.encode_images import HFStreamShardEncoder

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class TextAtlasStream:
    """Streams TextAtlas subsets sequentially without Arrow inference."""

    def __init__(self, subsets: List[str]):
        self.subsets = subsets

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for s in self.subsets:
            ds = load_dataset(
                "CSU-JPG/TextAtlas5M",
                s,
                split="train",
                streaming=True,
            )
            for ex in ds:
                ex["global_id"] = f"{s}:{ex.get('image_path', '')}"
                yield ex


class HFDataPipeline:
    """Coordinates stream joining, class balancing, and Hub uploads."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.hf_repo_id = self.config["hf_repo_id"]
        self.samples_per_source = self.config.get("samples_per_source", 100000)
        self.min_resolution = self.config.get("min_resolution", 512)
        self.avif_quality = self.config.get("avif_quality", 80)
        self.min_buffer_size = self.config.get("min_buffer_size", 512)
        self.max_synset_count = self.config.get("max_synset_count", 35)
        self.max_active_sources = self.config.get("max_active_sources", 3)

        hf_token = self.config.get("hf_token", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("tokenizer_name", "google/gemma-2b"),
            token=hf_token,
        )
        self.encoder = HFStreamShardEncoder(
            output_dir=self.config.get("temp_dir", "temp_shards"),
            hf_repo_id=self.hf_repo_id,
            samples_per_shard=self.config.get("samples_per_shard", 10000),
            length_tiers=self.config.get("length_tiers", [77, 152, 227]),
            hf_token=hf_token,
            in_channels=self.config.get("in_channels", 32),
            vae_downsample_factor=self.config.get("vae_downsample_factor", 8),
        )

        max_res_area = tuple(self.config.get("max_res_area", [768, 512]))
        base_res = tuple(self.config.get("base_res", [512, 512]))
        max_dim_limit = self.config.get("max_dim_limit", 1024)
        min_size = self.config.get("min_size", 256)
        divisible = self.config.get("divisible", 64)
        vae_factor = self.config.get("vae_downsample_factor", 8)
        max_log_ar_diff = self.config.get("max_log_ar_diff", 0.35)

        self.bucket_manager = OnlineBucketManager(
            max_res_area=max_res_area,
            base_res=base_res,
            max_dim_limit=max_dim_limit,
            min_resolution=self.min_resolution,
            min_size=min_size,
            divisible=divisible,
            vae_factor=vae_factor,
            max_log_ar_diff=max_log_ar_diff,
        )

    def _get_adapter_and_stream(
        self, ds_name: str
    ) -> Tuple[Optional[Any], Optional[BaseDatasetAdapter]]:
        """Instantiates stream and adapter with support for Megalith."""
        min_res = self.min_resolution
        quality = self.avif_quality
        if ds_name == "imagenet22k":
            ds = load_dataset("timm/imagenet-22k-wds", split="train", streaming=True)
            return ds, ImageNet22kWdsAdapter(ds_name, min_res, quality)
        if ds_name == "fluxreason":
            ds = load_dataset("LucasFang/FLUX-Reason-6M", split="train", streaming=True)
            return ds, FluxReasonAdapter(ds_name, min_res, quality)
        if ds_name == "gptedit":
            ds = load_dataset(
                "UCSC-VLAA/gpt-edit-simpler", split="train", streaming=True
            )
            return ds, GPTEditAdapter(ds_name, min_res, quality)
        if ds_name == "midjourneyv6":
            ds = load_dataset(
                "Photoroom/midjourney-v6-recap", split="train", streaming=True
            )
            return ds, MidjourneyV6Adapter(ds_name, min_res, quality)
        if ds_name == "pexels":
            ds = load_dataset(
                "animetimm/pexels-tagger-v0-w640-ws-full",
                split="train",
                streaming=True,
            )
            return ds, PexelsAdapter(ds_name, min_res, quality)
        if ds_name == "rendered_text":
            ds = load_dataset("wendlerc/RenderedText", split="train", streaming=True)
            return ds, RenderedTextAdapter(ds_name, min_res, quality)
        if ds_name == "megalith10m":
            ds = load_dataset(
                "drawthingsai/megalith-10m", split="train", streaming=True
            )
            return ds, MegalithAdapter(ds_name, min_res, quality)
        if ds_name == "textatlas":
            subsets = ["TextScenesHQ", "TextVisionBlend", "PPT2Structured"]
            return TextAtlasStream(subsets), TextAtlasAdapter(ds_name, min_res, quality)

        return None, None

    def _process_single_raw_sample(
        self,
        ds_name: str,
        item: Any,
        adapter: Optional[BaseDatasetAdapter],
        caption_map: StreamingI1CaptionMap,
        class_counts: Counter,
    ) -> Optional[Dict[str, Any]]:
        """Extracts and formats samples without monolithic tuple branch."""
        if adapter is None:
            return None
        res = adapter.extract_sample(item)
        if res is None:
            return None
        img_bytes, _, key, orig_w, orig_h, category = res

        # Balance classes across FLUX-Reason tasks and ImageNet synsets
        max_per_flux_cat = max(1, self.samples_per_source // 6)
        if ds_name == "fluxreason":
            if class_counts[category] >= max_per_flux_cat:
                return None
        elif ds_name == "imagenet22k":
            if category and class_counts[category] >= self.max_synset_count:
                return None

        if not key or key not in caption_map:
            return None
        captions = caption_map.get(key)
        if not captions:
            return None

        bucket_res = self.bucket_manager.assign_bucket(orig_w, orig_h)
        if bucket_res is None:
            return None
        bucket_idx, target_w, target_h, aspect_ratio = bucket_res

        tokens = self.tokenizer(captions[0], truncation=True, max_length=256)
        tier = self.encoder.determine_tier(len(tokens["input_ids"]))
        booru_id = f"{ds_name}_{key}"

        record = {
            "booru_id": booru_id,
            "image": img_bytes,
            "prompt": captions,
            "bucket_idx": bucket_idx,
            "target_width": target_w,
            "target_height": target_h,
            "original_width": orig_w,
            "original_height": orig_h,
            "aspect_ratio": aspect_ratio,
            "tier": tier,
            "aesthetic_tier": 3 if min(orig_w, orig_h) >= 1024 else 2,
            "tag_weight": 1.0,
        }
        class_counts[category] += 1
        return record

    def run_pipeline(self) -> None:
        """Runs sliding-window multi-source buffering to produce shards."""
        target_sources = list(
            self.config.get(
                "datasets",
                [
                    "imagenet22k",
                    "fluxreason",
                    "gptedit",
                    "midjourneyv6",
                    "pexels",
                    "rendered_text",
                    "textatlas",
                ],
            )
        )

        pending_sources = list(target_sources)
        random.shuffle(pending_sources)

        logger.info(
            f"Sliding pool configured: {len(pending_sources)} total sources, "
            f"max {self.max_active_sources} active concurrently."
        )

        streams: Dict[str, Iterator[Any]] = {}
        adapters: Dict[str, Optional[BaseDatasetAdapter]] = {}
        caption_maps: Dict[str, StreamingI1CaptionMap] = {}
        class_counters: Dict[str, Counter] = {s: Counter() for s in target_sources}
        extracted_counts: Dict[str, int] = {s: 0 for s in target_sources}
        stream_exhausted: Dict[str, bool] = {s: False for s in target_sources}
        buffers: Dict[str, deque] = {}
        active_sources: List[str] = []

        def _activate_source(source: str) -> bool:
            logger.info(f"Activating stream and caption loader for: '{source}'")
            caption_maps[source] = load_i1_caption_map(source)
            ds, ad = self._get_adapter_and_stream(source)
            if ds is not None and ad is not None:
                streams[source] = iter(ds)
                adapters[source] = ad
                buffers[source] = deque()
                active_sources.append(source)
                return True
            logger.warning(f"Failed to initialize stream for '{source}'.")
            return False

        def _retire_source(source: str) -> None:
            logger.info(
                f"Retiring source '{source}' "
                f"(extracted={extracted_counts[source]}, "
                f"exhausted={stream_exhausted[source]})."
            )
            if source in active_sources:
                active_sources.remove(source)
            streams.pop(source, None)
            adapters.pop(source, None)
            caption_maps.pop(source, None)
            buffers.pop(source, None)
            gc.collect()

        def _refill_source_buffer(source: str) -> None:
            if stream_exhausted[source]:
                return
            stream_iter = streams[source]
            adapter = adapters[source]
            c_map = caption_maps[source]
            c_counts = class_counters[source]

            while (
                len(buffers[source]) < self.min_buffer_size
                and extracted_counts[source] < self.samples_per_source
            ):
                try:
                    raw_item = next(stream_iter)
                except StopIteration:
                    stream_exhausted[source] = True
                    break
                rec = self._process_single_raw_sample(
                    source, raw_item, adapter, c_map, c_counts
                )
                if rec is not None:
                    buffers[source].append(rec)
                    extracted_counts[source] += 1

        # Populate initial active window
        while len(active_sources) < self.max_active_sources and pending_sources:
            _activate_source(pending_sources.pop(0))

        shard_idx = 0
        total_processed = 0
        shard_metadata: List[Dict[str, Any]] = []
        samples_per_shard = self.encoder.samples_per_shard

        while active_sources:
            current_records: List[Dict[str, Any]] = []

            # Refill buffers and retire exhausted sources whose buffers drained
            for ds_name in list(active_sources):
                _refill_source_buffer(ds_name)
                source_finished = (
                    stream_exhausted[ds_name]
                    or extracted_counts[ds_name] >= self.samples_per_source
                )
                if not buffers[ds_name] and source_finished:
                    _retire_source(ds_name)
                    # Pull next dataset into active window
                    while (
                        len(active_sources) < self.max_active_sources
                        and pending_sources
                    ):
                        next_ds = pending_sources.pop(0)
                        if _activate_source(next_ds):
                            _refill_source_buffer(next_ds)

            if not active_sources and not any(len(b) > 0 for b in buffers.values()):
                break

            target_slice = max(1, samples_per_shard // max(1, len(active_sources)))

            # Balanced extraction from active pool
            for ds_name in list(active_sources):
                buf = buffers[ds_name]
                take = min(len(buf), target_slice)
                for _ in range(take):
                    current_records.append(buf.popleft())

            needed = samples_per_shard - len(current_records)
            if needed > 0:
                for ds_name in active_sources:
                    buf = buffers[ds_name]
                    take = min(len(buf), needed)
                    for _ in range(take):
                        current_records.append(buf.popleft())
                    needed -= take
                    if needed <= 0:
                        break

            if not current_records:
                break

            random.shuffle(current_records)

            summary = self.encoder.write_upload_and_clean_shard(
                current_records, shard_idx
            )
            shard_metadata.append(summary)
            total_processed += len(current_records)
            shard_idx += 1
            logger.info(
                f"Shard {shard_idx - 1} assembled with {len(current_records)} "
                f"samples across {len(active_sources)} active sources."
            )

        self.encoder.finalize_metadata(
            shard_metadata,
            total_processed,
            self.bucket_manager.bucket_resolutions,
        )
        logger.info(
            f"Sliding curation complete: {total_processed} samples "
            f"uploaded across {len(shard_metadata)} mixed shards."
        )
