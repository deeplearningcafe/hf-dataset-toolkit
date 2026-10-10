import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class HFStreamShardEncoder:
    """Encodes streaming samples, manages Hub uploads, and cleans disk."""

    def __init__(
        self,
        output_dir: str,
        hf_repo_id: str,
        samples_per_shard: int = 5000,
        length_tiers: Optional[List[int]] = None,
        parquet_compression: str = "SNAPPY",
        hf_token: Optional[str] = None,
        in_channels: int = 32,
        vae_downsample_factor: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.hf_repo_id = hf_repo_id
        self.samples_per_shard = samples_per_shard
        self.length_tiers = sorted(length_tiers or [77, 152, 227])
        self.max_length = self.length_tiers[-1]
        self.parquet_compression = parquet_compression
        self.in_channels = in_channels
        self.vae_downsample_factor = vae_downsample_factor
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api = HfApi(token=hf_token)
        self.api.create_repo(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

        self.schema = pa.schema(
            [
                ("booru_id", pa.string()),
                ("image", pa.binary()),
                ("prompt", pa.list_(pa.string())),
                ("bucket_idx", pa.int32()),
                ("target_width", pa.int32()),
                ("target_height", pa.int32()),
                ("original_width", pa.int32()),
                ("original_height", pa.int32()),
                ("aspect_ratio", pa.float32()),
                ("tier", pa.int32()),
                ("aesthetic_tier", pa.int32()),
                ("tag_weight", pa.float32()),
            ]
        )

    def get_shard_dir(self, shard_idx: int) -> Path:
        """Returns and creates a dedicated scratch subdirectory for a shard."""
        shard_dir = self.output_dir / f"shard_{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir

    def determine_tier(self, token_len: int) -> int:
        """Determines token sequence tier."""
        for tier in self.length_tiers:
            if token_len <= tier:
                return tier
        return self.max_length

    def write_upload_and_clean_shard(
        self, records: List[Dict[str, Any]], shard_idx: int
    ) -> Dict[str, Any]:
        """Saves shard locally, uploads to Hub, and purges the shard folder."""
        shard_dir = self.get_shard_dir(shard_idx)
        fname = f"data_shard_{shard_idx:05d}.parquet"
        shard_path = shard_dir / fname

        pydict = {col: [r[col] for r in records] for col in self.schema.names}
        table = pa.Table.from_pydict(pydict, schema=self.schema)

        pq.write_table(
            table,
            shard_path,
            compression=self.parquet_compression,
            use_dictionary=True,
        )

        max_retries = 5
        uploaded = False
        for attempt in range(max_retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(shard_path),
                    path_in_repo=fname,
                    repo_id=self.hf_repo_id,
                    repo_type="dataset",
                )
                uploaded = True
                break
            except Exception as e:
                wait_time = 2**attempt
                logger.warning(
                    f"Upload attempt {attempt + 1} failed ({e}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        if not uploaded:
            raise RuntimeError(f"Failed to upload {fname} after retries.")

        try:
            shutil.rmtree(shard_dir)
            logger.info(f"Uploaded {fname} and removed shard folder: {shard_dir.name}")
        except Exception as e:
            logger.warning(f"Could not remove shard directory {shard_dir}: {e}")

        count_gte_1024 = sum(
            1
            for r in records
            if not str(r["booru_id"]).startswith("rendered_text_")
            and min(r["original_width"], r["original_height"]) >= 1024
        )
        aes_counts = {
            2: sum(1 for r in records if r["aesthetic_tier"] == 2),
            3: sum(1 for r in records if r["aesthetic_tier"] == 3),
        }

        return {
            "shard_file": fname,
            "sample_count": len(records),
            "count_gte_1024": count_gte_1024,
            "aesthetic_tier_counts": aes_counts,
        }

    def finalize_metadata(
        self,
        shard_metadata: List[Dict[str, Any]],
        total_samples: int,
        bucket_resolutions: List[Tuple[int, int]],
    ) -> None:
        """Pushes aggregated metadata.json manifest to the Hub repository."""
        total_gte_1024 = sum(s.get("count_gte_1024", 0) for s in shard_metadata)
        total_tier_2 = sum(
            s.get("aesthetic_tier_counts", {}).get(2, 0) for s in shard_metadata
        )
        total_tier_3 = sum(
            s.get("aesthetic_tier_counts", {}).get(3, 0) for s in shard_metadata
        )

        meta_dict = {
            "total_samples": total_samples,
            "total_samples_gte_1024": total_gte_1024,
            "aesthetic_tier_counts": {
                "2": total_tier_2,
                "3": total_tier_3,
            },
            "num_shards": len(shard_metadata),
            "samples_per_shard": self.samples_per_shard,
            "length_tiers": self.length_tiers,
            "shards": shard_metadata,
            "bucket_info": [
                {
                    "bucket_idx": i,
                    "resolution": list(res),
                    "latents_resolution": [
                        res[1] // self.vae_downsample_factor,
                        res[0] // self.vae_downsample_factor,
                    ],
                    "latent_channels": self.in_channels,
                    "log_aspect_ratio": round(math.log(res[0] / max(1, res[1])), 4),
                }
                for i, res in enumerate(bucket_resolutions)
            ],
        }

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=4)

        self.api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="metadata.json",
            repo_id=self.hf_repo_id,
            repo_type="dataset",
        )
        os.remove(meta_path)
        logger.info("Aligned metadata.json successfully uploaded.")
