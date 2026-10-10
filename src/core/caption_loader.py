"""Zero-disk HTTP streaming loader for i1-captions with lookahead caching."""

import logging
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Sequence, Any
from datasets import load_dataset

logger = logging.getLogger(__name__)


class StreamingI1CaptionMap:
    """Sliding-window stream reader matching image IDs without disk bloat."""

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
    ):
        self.subset = subset
        self.caption_columns = caption_columns
        self.max_cache_size = max_cache_size
        self.lookahead_step = lookahead_step
        self.cache: OrderedDict[str, List[str]] = OrderedDict()
        self.stream_exhausted = False

        logger.info(f"Initializing streaming caption reader for subset '{subset}'...")
        self.dataset = load_dataset(
            "zlab-princeton/i1-captions",
            subset,
            split="train",
            streaming=True,
        )
        self.iterator: Iterator[Dict] = iter(self.dataset)

    def _advance_stream(self, target_key: Optional[str] = None) -> bool:
        """Pulls streaming items until target_key is found or step limit hit."""
        if self.stream_exhausted:
            return False

        steps = 0
        while steps < self.lookahead_step:
            try:
                row = next(self.iterator)
                steps += 1
            except StopIteration:
                self.stream_exhausted = True
                return False

            key = str(row.get("key", ""))
            captions = [
                str(row[c])
                for c in self.caption_columns
                if c in row and row[c] is not None
            ]
            if key and captions:
                self.cache[key] = captions
                if len(self.cache) > self.max_cache_size:
                    self.cache.popitem(last=False)
                if target_key is not None and key == target_key:
                    return True

        return target_key in self.cache if target_key else True

    def get(self, key: str, default: Any = None) -> Any:
        """Look up captions by key, fetching ahead if necessary."""
        str_key = str(key)
        if str_key in self.cache:
            return self.cache[str_key]
        if not self.stream_exhausted:
            if self._advance_stream(target_key=str_key):
                return self.cache[str_key]
        return default

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> List[str]:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Key '{key}' not found in {self.subset} stream.")
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
) -> StreamingI1CaptionMap:
    """Backward-compatible factory yielding an on-demand streaming map."""
    return StreamingI1CaptionMap(subset, caption_columns=caption_columns)
