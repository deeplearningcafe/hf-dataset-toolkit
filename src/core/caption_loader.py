import bisect
import fnmatch
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class StreamingI1CaptionMap:
    """Indexed on-disk Parquet caption map with active Row Group caching.

    Keeps only keys in memory (~75MB) and reads individual Row Groups on-demand
    (~5MB) from the local shard file, eliminating OOM crashes.
    """

    def __init__(
        self,
        subset: str,
        caption_columns: Sequence[str] = (
            "caption1",
            "caption2",
            "caption3",
            "caption4",
            "caption5",
        ),
        max_cache_size: int = 50000,
        lookahead_step: int = 20000,
        hf_token: Optional[str] = None,
        temp_dir: str = "temp_shards",
    ):
        self.subset = subset
        self.caption_columns = list(caption_columns)
        self.hf_token = hf_token
        self.scratch_dir = Path(temp_dir) / "captions" / subset
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self.api = HfApi(token=hf_token)
        self.repo_id = "zlab-princeton/i1-captions"
        self.shard_files: List[str] = self._discover_shard_files()
        self.current_shard_idx: int = -1

        self.local_shard_path: Optional[str] = None
        self.pq_file: Optional[pq.ParquetFile] = None
        self.key_to_row: Dict[str, int] = {}
        self.rg_starts: List[int] = []
        self.cached_rg_idx: int = -1
        self.cached_rg_dict: Dict[str, List[str]] = {}

        logger.info(
            f"Initialized low-memory caption map for '{subset}'. "
            f"Found {len(self.shard_files)} shards in {self.repo_id}."
        )
        if self.shard_files:
            self._load_shard(0)

    def _discover_shard_files(self) -> List[str]:
        all_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        pattern = f"{self.subset}/*.parquet"
        matched = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
        return sorted(matched)

    def _cleanup_current_shard(self) -> None:
        """Closes parquet handle and removes local shard file from disk."""
        self.pq_file = None
        self.cached_rg_dict.clear()
        self.key_to_row.clear()
        self.rg_starts.clear()
        self.cached_rg_idx = -1

        if self.local_shard_path and os.path.exists(self.local_shard_path):
            try:
                os.remove(self.local_shard_path)
                logger.info(f"Removed local caption shard: {self.local_shard_path}")
            except OSError as e:
                logger.warning(
                    f"Could not purge caption file {self.local_shard_path}: {e}"
                )
        self.local_shard_path = None

    def _load_shard(self, shard_idx: int) -> bool:
        """Downloads shard and indexes only keys into memory."""
        if shard_idx < 0 or shard_idx >= len(self.shard_files):
            return False

        self._cleanup_current_shard()
        shard_rel_path = self.shard_files[shard_idx]
        logger.info(
            f"Downloading caption shard {shard_idx + 1}/"
            f"{len(self.shard_files)}: {shard_rel_path}"
        )
        self.local_shard_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_rel_path,
            repo_type="dataset",
            token=self.hf_token,
            local_dir=str(self.scratch_dir),
        )

        self.pq_file = pq.ParquetFile(self.local_shard_path)

        # Index row group boundaries for binary search
        cum = 0
        self.rg_starts = []
        for rg in range(self.pq_file.num_row_groups):
            self.rg_starts.append(cum)
            cum += self.pq_file.metadata.row_group(rg).num_rows

        # Load 'key' column
        key_batch = self.pq_file.read(columns=["key"])
        keys_list = key_batch["key"].to_pylist()
        self.key_to_row = {str(k): i for i, k in enumerate(keys_list)}
        del key_batch
        del keys_list

        self.current_shard_idx = shard_idx
        logger.info(
            f"Indexed {len(self.key_to_row)} keys from caption shard "
            f"{shard_idx} into memory."
        )
        return True

    def _fetch_from_disk(self, row_idx: int) -> Optional[List[str]]:
        """Reads row from cached row group or fetches row group from disk."""
        if self.pq_file is None:
            return None

        # Binary search for the row group containing row_idx
        rg_idx = bisect.bisect_right(self.rg_starts, row_idx) - 1
        offset_in_rg = row_idx - self.rg_starts[rg_idx]

        if self.cached_rg_idx != rg_idx:
            # Read only the single active Row Group
            available_cols = [
                c for c in self.caption_columns if c in self.pq_file.schema.names
            ]
            rg_table = self.pq_file.read_row_group(rg_idx, columns=available_cols)
            rg_pydict = rg_table.to_pydict()
            num_rows = len(next(iter(rg_pydict.values())))

            new_rg_cache = {}
            for i in range(num_rows):
                caps = [
                    str(rg_pydict[col][i])
                    for col in available_cols
                    if rg_pydict[col][i] is not None
                ]
                if caps:
                    new_rg_cache[i] = caps

            self.cached_rg_dict = new_rg_cache
            self.cached_rg_idx = rg_idx

        return self.cached_rg_dict.get(offset_in_rg)

    def get(self, key: str, default: Any = None) -> Any:
        """Looks up captions by key, pulling from next shard if necessary."""
        str_key = str(key)
        if str_key in self.key_to_row:
            res = self._fetch_from_disk(self.key_to_row[str_key])
            if res is not None:
                return res

        next_idx = self.current_shard_idx + 1
        while next_idx < len(self.shard_files):
            if self._load_shard(next_idx):
                if str_key in self.key_to_row:
                    res = self._fetch_from_disk(self.key_to_row[str_key])
                    if res is not None:
                        return res
            next_idx += 1

        return default

    def close(self) -> None:
        """Explicitly releases file handles and purges temporary files."""
        self._cleanup_current_shard()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> List[str]:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Key '{key}' not found in {self.subset} caption shards.")
        return val


def load_i1_caption_map(
    subset: str,
    caption_columns: Sequence[str] = (
        "caption1",
        "caption2",
        "caption3",
        "caption4",
        "caption5",
    ),
    num_proc: int = 4,
    hf_token: Optional[str] = None,
    temp_dir: str = "temp_shards",
) -> StreamingI1CaptionMap:
    """Backward-compatible factory yielding low-memory shard caption map."""
    return StreamingI1CaptionMap(
        subset=subset,
        caption_columns=caption_columns,
        hf_token=hf_token,
        temp_dir=temp_dir,
    )
