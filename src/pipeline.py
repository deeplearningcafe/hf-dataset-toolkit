from collections import Counter, deque
import gc
import logging
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml
import sys
import fnmatch
import os
from pathlib import Path
import tarfile
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from concurrent.futures import Future, ThreadPoolExecutor
import time
from tqdm.auto import tqdm


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


class HFShardStreamReader:
    """Streams dataset records by downloading and purging shards one-by-one."""

    def __init__(
        self,
        repo_id: str,
        patterns: List[str],
        format_type: str,
        temp_dir: str = "temp_shards",
        hf_token: Optional[str] = None,
        max_prefetch: int = 1,
    ):
        self.repo_id = repo_id
        self.patterns = patterns
        self.format_type = format_type
        self.scratch_dir = Path(temp_dir) / "downloads" / repo_id.replace("/", "_")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.max_prefetch = max(0, max_prefetch)

        self.api = HfApi(token=hf_token)
        self.shard_files = self._discover_files()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futures: Dict[int, Future[str]] = {}
        self._current_local_path: Optional[str] = None

        logger.info(
            f"HFShardStreamReader: '{repo_id}' ({format_type}) initialized "
            f"with {len(self.shard_files)} shards (prefetch={max_prefetch})."
        )

    def _discover_files(self) -> List[str]:
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        matched = []
        for pat in self.patterns:
            matched.extend([f for f in all_files if fnmatch.fnmatch(f, pat)])
        return sorted(list(set(matched)))

    def _download_shard(self, shard_file: str) -> str:
        """Downloads single shard file with duration and size tracking."""
        t0 = time.time()
        logger.info(f"Downloading shard: {self.repo_id} -> {shard_file}")
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_file,
            repo_type="dataset",
            token=self.hf_token,
            local_dir=str(self.scratch_dir),
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        speed = size_mb / max(0.001, elapsed)
        logger.info(
            f"Fetched {shard_file} ({size_mb:.1f} MB) in {elapsed:.1f}s "
            f"({speed:.1f} MB/s)"
        )
        return local_path

    def _iter_parquet_shard(
        self, shard_path: str, shard_name: str
    ) -> Iterator[Dict[str, Any]]:
        pq_file = pq.ParquetFile(shard_path)
        for batch in pq_file.iter_batches(batch_size=1024):
            cols = batch.to_pydict()
            col_names = list(cols.keys())
            if not col_names:
                continue
            num_rows = len(cols[col_names[0]])
            for i in range(num_rows):
                row = {c: cols[c][i] for c in col_names}
                if "global_id" not in row and "image_path" in row:
                    subset = shard_name.split("/")[0] if "/" in shard_name else ""
                    if subset:
                        row["global_id"] = f"{subset}:{row['image_path']}"
                yield row

    def _iter_tar_shard(self, shard_path: str) -> Iterator[Dict[str, Any]]:
        """Parses tar members while skipping hidden/AppleDouble dotfiles."""
        try:
            with tarfile.open(shard_path, "r:*") as tar:
                current_key = None
                current_sample: Dict[str, Any] = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    fname = os.path.basename(member.name)
                    # Ignore AppleDouble or OS hidden files (e.g. ._123.webp)
                    if fname.startswith("."):
                        continue
                    parts = fname.rsplit(".", 1)
                    if len(parts) != 2:
                        continue
                    key, ext = parts[0], parts[1].lower()
                    if current_key is None:
                        current_key = key
                    elif key != current_key:
                        if current_sample:
                            current_sample["__key__"] = current_key
                            yield current_sample
                        current_key = key
                        current_sample = {}

                    try:
                        extracted = tar.extractfile(member)
                        if extracted is not None:
                            current_sample[ext] = extracted.read()
                    except Exception as e:
                        logger.warning(
                            f"Failed extracting member {member.name} in "
                            f"{shard_path}: {e}"
                        )

                if current_sample and current_key:
                    current_sample["__key__"] = current_key
                    yield current_sample
        except Exception as e:
            logger.error(
                f"Failed reading tar archive {shard_path}: {e}",
                exc_info=True,
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Double-buffered generator overlapping disk read and net download."""
        if not self.shard_files:
            logger.warning(f"No shards found for repo '{self.repo_id}'.")
            return

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_futures.clear()

        def _schedule_download(idx: int) -> None:
            if idx < len(self.shard_files) and idx not in self._prefetch_futures:
                shard_f = self.shard_files[idx]
                self._prefetch_futures[idx] = self._executor.submit(
                    self._download_shard, shard_f
                )

        # Prefetch initial shards up to bounded window (current + prefetch)
        for i in range(min(1 + self.max_prefetch, len(self.shard_files))):
            _schedule_download(i)

        try:
            for idx, shard_file in enumerate(self.shard_files):
                _schedule_download(idx + 1)
                future = self._prefetch_futures.pop(idx, None)
                if future is None:
                    future = self._executor.submit(self._download_shard, shard_file)

                try:
                    self._current_local_path = future.result()
                except Exception as e:
                    logger.error(
                        f"Failed downloading {shard_file} from {self.repo_id}: {e}",
                        exc_info=True,
                    )
                    continue

                shard_count = 0
                try:
                    if self.format_type == "parquet":
                        for item in self._iter_parquet_shard(
                            self._current_local_path, shard_file
                        ):
                            shard_count += 1
                            yield item
                    elif self.format_type == "tar":
                        for item in self._iter_tar_shard(self._current_local_path):
                            shard_count += 1
                            yield item
                    else:
                        raise ValueError(f"Unsupported format: {self.format_type}")
                    logger.info(f"Completed {shard_file} ({shard_count} samples)")
                except Exception as e:
                    logger.error(
                        f"Error reading {shard_file} ({self.repo_id}): {e}",
                        exc_info=True,
                    )
                finally:
                    if self._current_local_path and os.path.exists(
                        self._current_local_path
                    ):
                        try:
                            os.remove(self._current_local_path)
                        except OSError as e:
                            logger.warning(
                                f"Failed removing {self._current_local_path}: {e}"
                            )
                        self._current_local_path = None
        finally:
            self.close()

    def close(self) -> None:
        """Cancels pending prefetch futures and purges local files from disk."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

        for fut in list(self._prefetch_futures.values()):
            if fut.done() and not fut.cancelled() and not fut.exception():
                try:
                    path = fut.result()
                    if path and os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Purged unused prefetched shard: {path}")
                except Exception:
                    pass
        self._prefetch_futures.clear()

        if self._current_local_path and os.path.exists(self._current_local_path):
            try:
                os.remove(self._current_local_path)
            except OSError:
                pass
            self._current_local_path = None


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

        self.hf_token = self.config.get("hf_token", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("tokenizer_name", "google/gemma-2b"),
            token=self.hf_token,
        )
        self.encoder = HFStreamShardEncoder(
            output_dir=self.config.get("temp_dir", "temp_shards"),
            hf_repo_id=self.hf_repo_id,
            samples_per_shard=self.config.get("samples_per_shard", 10000),
            length_tiers=self.config.get("length_tiers", [77, 152, 227]),
            hf_token=self.hf_token,
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
        self.skip_counters: Dict[str, Counter] = {}

    def _get_adapter_and_stream(
        self, ds_name: str
    ) -> Tuple[Optional[Any], Optional[BaseDatasetAdapter]]:
        """Instantiates automated JIT shard reader and corresponding adapter."""
        min_res = self.min_resolution
        quality = self.avif_quality
        token = self.hf_token
        temp = self.config.get("temp_dir", "temp_shards")
        prefetch = self.config.get("max_prefetch", 1)

        if ds_name == "imagenet22k":
            reader = HFShardStreamReader(
                repo_id="timm/imagenet-22k-wds",
                patterns=["imagenet22k-train-*.tar"],
                format_type="tar",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, ImageNet22kWdsAdapter(ds_name, min_res, quality)

        if ds_name == "fluxreason":
            reader = HFShardStreamReader(
                repo_id="LucasFang/FLUX-Reason-6M",
                patterns=["Aesthetics-Part*/*.parquet", "*.parquet"],
                format_type="parquet",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, FluxReasonAdapter(ds_name, min_res, quality)

        if ds_name == "gptedit":
            reader = HFShardStreamReader(
                repo_id="UCSC-VLAA/gpt-edit-simpler",
                patterns=["*.parquet", "data/*.parquet"],
                format_type="parquet",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, GPTEditAdapter(ds_name, min_res, quality)

        if ds_name == "midjourneyv6":
            reader = HFShardStreamReader(
                repo_id="Photoroom/midjourney-v6-recap",
                patterns=["*.parquet", "default/train/*.parquet"],
                format_type="parquet",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, MidjourneyV6Adapter(ds_name, min_res, quality)

        if ds_name == "pexels":
            reader = HFShardStreamReader(
                repo_id="animetimm/pexels-tagger-v0-w640-ws-full",
                patterns=["train/*.tar", "train-*.tar"],
                format_type="tar",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, PexelsAdapter(ds_name, min_res, quality)

        if ds_name == "rendered_text":
            reader = HFShardStreamReader(
                repo_id="wendlerc/RenderedText",
                patterns=["*.tar", "remaining/*.tar"],
                format_type="tar",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, RenderedTextAdapter(ds_name, min_res, quality)

        if ds_name == "megalith10m":
            reader = HFShardStreamReader(
                repo_id="drawthingsai/megalith-10m",
                patterns=["*.tar"],
                format_type="tar",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, MegalithAdapter(ds_name, min_res, quality)

        if ds_name == "textatlas":
            reader = HFShardStreamReader(
                repo_id="CSU-JPG/TextAtlas5M",
                patterns=[
                    "TextScenesHQ/*.parquet",
                    "TextVisionBlend/*.parquet",
                    "PPT2Structured/*.parquet",
                ],
                format_type="parquet",
                temp_dir=temp,
                hf_token=token,
                max_prefetch=prefetch,
            )
            return reader, TextAtlasAdapter(ds_name, min_res, quality)

        return None, None

    def _process_single_raw_sample(
        self,
        ds_name: str,
        item: Any,
        adapter: Optional[BaseDatasetAdapter],
        caption_map: StreamingI1CaptionMap,
        class_counts: Counter,
    ) -> Optional[Dict[str, Any]]:
        """Extracts and formats samples with real-time skip diagnostics."""
        if adapter is None:
            return None
        res = adapter.extract_sample(item)
        if res is None:
            self.skip_counters[ds_name]["extract_or_res"] += 1
            return None
        img_bytes, _, key, orig_w, orig_h, category = res

        # Configurable FLUX category cap (default 40% of samples_per_source)
        # flux_cap = self.config.get(
        #     "max_flux_category_count",
        #     max(1, int(self.samples_per_source * 0.40)),
        # )
        # if ds_name == "fluxreason":
        #     if category != "general" and class_counts[category] >= flux_cap:
        #         self.skip_counters[ds_name]["category_cap"] += 1
        #         return None
        # elif ds_name == "imagenet22k":
        #     if category and class_counts[category] >= self.max_synset_count:
        #         self.skip_counters[ds_name]["synset_cap"] += 1
        #         return None

        if not key or key not in caption_map:
            self.skip_counters[ds_name]["no_caption"] += 1
            return None
        captions = caption_map.get(key)
        if not captions:
            self.skip_counters[ds_name]["empty_caption"] += 1
            return None

        bucket_res = self.bucket_manager.assign_bucket(orig_w, orig_h)
        if bucket_res is None:
            self.skip_counters[ds_name]["bucket_mismatch"] += 1
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
        self.skip_counters = {s: Counter() for s in target_sources}
        buffers: Dict[str, deque] = {}
        active_sources: List[str] = []

        def _activate_source(source: str) -> bool:
            logger.info(f"Activating stream and caption loader for: '{source}'")
            caption_maps[source] = load_i1_caption_map(
                source,
                hf_token=self.hf_token,
                temp_dir=self.config.get("temp_dir", "temp_shards"),
            )
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
                f"exhausted={stream_exhausted[source]}, "
                f"skips={dict(self.skip_counters[source])})."
            )
            if source in active_sources:
                active_sources.remove(source)

            ds_stream = streams.pop(source, None)
            if ds_stream is not None and hasattr(ds_stream, "close"):
                ds_stream.close()

            adapters.pop(source, None)

            c_map = caption_maps.pop(source, None)
            if c_map is not None:
                c_map.close()

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
                except Exception as e:
                    logger.error(
                        f"Stream error for source '{source}': {e}",
                        exc_info=True,
                    )
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

        while active_sources or any(len(b) > 0 for b in buffers.values()):
            current_records: List[Dict[str, Any]] = []

            # Progress bar tracking current shard assembly
            pbar = tqdm(
                total=samples_per_shard,
                desc=f"Shard {shard_idx:05d}",
                unit="img",
                dynamic_ncols=True,
                leave=False,
            )

            while len(current_records) < samples_per_shard:
                progress_made = False

                if active_sources:
                    with ThreadPoolExecutor(max_workers=len(active_sources)) as pool:
                        list(pool.map(_refill_source_buffer, active_sources))

                for ds_name in list(active_sources):
                    buf = buffers[ds_name]

                    if buf:
                        needed = samples_per_shard - len(current_records)
                        take = min(len(buf), needed)
                        for _ in range(take):
                            current_records.append(buf.popleft())

                        pbar.update(take)
                        pbar.set_postfix(
                            {
                                ds: f"{extracted_counts[ds]}/{self.samples_per_source}"
                                for ds in active_sources
                            }
                        )
                        progress_made = True

                    source_finished = (
                        stream_exhausted[ds_name]
                        or extracted_counts[ds_name] >= self.samples_per_source
                    )
                    if not buf and source_finished:
                        _retire_source(ds_name)
                        while (
                            len(active_sources) < self.max_active_sources
                            and pending_sources
                        ):
                            next_ds = pending_sources.pop(0)
                            if _activate_source(next_ds):
                                _refill_source_buffer(next_ds)

                    if len(current_records) >= samples_per_shard:
                        break

                if not progress_made:
                    break

            pbar.close()

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
