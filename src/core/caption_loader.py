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
    """Indexed Parquet caption map with multi-shard retention and negative caching.

    Maintains up to max_active_shards locally, prevents monotonic forward-only
    purging, and avoids catastrophic cache destruction when shards are shuffled.
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
        max_active_shards: int = 3,
    ):
        self.subset = subset
        self.caption_columns = list(caption_columns)
        self.hf_token = hf_token
        self.scratch_dir = Path(temp_dir) / "captions" / subset
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.max_active_shards = max(1, max_active_shards)

        self.api = HfApi(token=hf_token)
        self.repo_id = "zlab-princeton/i1-captions"
        self.shard_files: List[str] = self._discover_shard_files()

        self.active_shard_indices: List[int] = []
        self.local_shard_paths: Dict[int, str] = {}
        self.pq_files: Dict[int, pq.ParquetFile] = {}
        self.key_to_row_maps: Dict[int, Dict[str, int]] = {}
        self.rg_starts_maps: Dict[int, List[int]] = {}

        self.key_to_shard: Dict[str, int] = {}
        self.indexed_shards = set()

        self.cached_rg_idx: Dict[int, int] = {}
        self.cached_rg_dict: Dict[int, Dict[str, List[str]]] = {}
        self.negative_key_cache = set()

        logger.info(
            f"Initialized caption map for '{subset}' "
            f"({len(self.shard_files)} shards found, retention="
            f"{self.max_active_shards})."
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

    def _purge_oldest_shard(self) -> None:
        """Evicts oldest loaded in-memory handle without wiping disk files."""
        if len(self.active_shard_indices) <= self.max_active_shards:
            return

        evict_idx = self.active_shard_indices.pop(0)
        self.pq_files.pop(evict_idx, None)
        self.key_to_row_maps.pop(evict_idx, None)
        self.rg_starts_maps.pop(evict_idx, None)
        self.cached_rg_idx.pop(evict_idx, None)
        self.cached_rg_dict.pop(evict_idx, None)

    def _load_shard(self, shard_idx: int) -> bool:
        """Downloads shard and indexes key column into memory."""
        if shard_idx < 0 or shard_idx >= len(self.shard_files):
            return False

        if shard_idx in self.pq_files:
            if shard_idx in self.active_shard_indices:
                self.active_shard_indices.remove(shard_idx)
            self.active_shard_indices.append(shard_idx)
            return True

        shard_rel_path = self.shard_files[shard_idx]
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_rel_path,
            repo_type="dataset",
            token=self.hf_token,
            local_dir=str(self.scratch_dir),
        )

        pq_file = pq.ParquetFile(local_path)

        cum = 0
        rg_starts = []
        for rg in range(pq_file.num_row_groups):
            rg_starts.append(cum)
            cum += pq_file.metadata.row_group(rg).num_rows

        key_batch = pq_file.read(columns=["key"])
        keys_list = key_batch["key"].to_pylist()
        key_to_row = {str(k): i for i, k in enumerate(keys_list)}
        del key_batch

        # NEW: Register keys in global shard index to avoid searching blind
        for k in key_to_row:
            self.key_to_shard[k] = shard_idx
        self.indexed_shards.add(shard_idx)

        self.local_shard_paths[shard_idx] = local_path
        self.pq_files[shard_idx] = pq_file
        self.rg_starts_maps[shard_idx] = rg_starts
        self.key_to_row_maps[shard_idx] = key_to_row
        self.cached_rg_idx[shard_idx] = -1
        self.cached_rg_dict[shard_idx] = {}
        self.active_shard_indices.append(shard_idx)

        self._purge_oldest_shard()
        return True

    def _fetch_from_disk(self, shard_idx: int, row_idx: int) -> Optional[List[str]]:
        """Reads row from cached row group or fetches row group from disk."""
        pq_file = self.pq_files.get(shard_idx)
        rg_starts = self.rg_starts_maps.get(shard_idx)
        if pq_file is None or rg_starts is None:
            return None

        rg_idx = bisect.bisect_right(rg_starts, row_idx) - 1
        offset_in_rg = row_idx - rg_starts[rg_idx]

        if self.cached_rg_idx.get(shard_idx) != rg_idx:
            available_cols = [
                c for c in self.caption_columns if c in pq_file.schema.names
            ]
            rg_table = pq_file.read_row_group(rg_idx, columns=available_cols)
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

            self.cached_rg_dict[shard_idx] = new_rg_cache
            self.cached_rg_idx[shard_idx] = rg_idx

        return self.cached_rg_dict[shard_idx].get(offset_in_rg)

    def get(self, key: str, default: Any = None) -> Any:
        """Looks up captions across retained shards with indexed routing."""
        str_key = str(key)
        if str_key in self.negative_key_cache:
            return default

        # 1. Direct hit if shard location is known
        if str_key in self.key_to_shard:
            target_shard = self.key_to_shard[str_key]
            if target_shard not in self.pq_files:
                self._load_shard(target_shard)
            row_idx = self.key_to_row_maps[target_shard][str_key]
            res = self._fetch_from_disk(target_shard, row_idx)
            if res is not None:
                return res

        # 2. Search unindexed shards sequentially if key has not been seen
        unindexed = [
            i for i in range(len(self.shard_files)) if i not in self.indexed_shards
        ]
        for shard_idx in unindexed:
            if self._load_shard(shard_idx):
                if str_key in self.key_to_shard:
                    row_idx = self.key_to_row_maps[shard_idx][str_key]
                    res = self._fetch_from_disk(shard_idx, row_idx)
                    if res is not None:
                        return res

        if len(self.negative_key_cache) > 200000:
            self.negative_key_cache.clear()
        self.negative_key_cache.add(str_key)
        return default

    def close(self) -> None:
        """Purges handles and removes all local cached caption shards."""
        for path in list(self.local_shard_paths.values()):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        self.local_shard_paths.clear()
        self.pq_files.clear()
        self.key_to_row_maps.clear()
        self.rg_starts_maps.clear()
        self.cached_rg_dict.clear()
        self.negative_key_cache.clear()

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
    max_active_shards: int = 3,
) -> StreamingI1CaptionMap:
    """Backward-compatible factory yielding low-memory shard caption map."""
    return StreamingI1CaptionMap(
        subset=subset,
        caption_columns=caption_columns,
        hf_token=hf_token,
        temp_dir=temp_dir,
        max_active_shards=max_active_shards,
    )
