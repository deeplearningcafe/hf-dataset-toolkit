import math
from typing import Any, Dict, List, Optional, Set, Tuple

Resolution = Tuple[int, int]


class OnlineBucketManager:
    """Computes aspect ratio buckets online using NovelAI token constraints."""

    def __init__(
        self,
        max_res_area: Tuple[int, int] = (768, 512),
        base_res: Tuple[int, int] = (512, 512),
        max_dim_limit: int = 1024,
        min_resolution: int = 512,
        min_size: int = 256,
        divisible: int = 64,
        vae_factor: int = 8,
        max_log_ar_diff: float = 0.35,
    ):
        self.max_res_area = max_res_area
        self.base_res = base_res
        self.max_dim_limit = max_dim_limit
        self.min_resolution = min_resolution
        self.min_size = min_size
        self.divisible = divisible
        self.vae_factor = vae_factor
        self.max_log_ar_diff = max_log_ar_diff

        (
            self.bucket_resolutions,
            self.log_aspect_ratios,
        ) = self._generate_novelai_buckets()

    def _generate_novelai_buckets(
        self,
    ) -> Tuple[List[Resolution], List[float]]:
        """Generates bucket resolutions based on NovelAI max latent tokens."""
        max_tokens = (self.max_res_area[0] / self.vae_factor) * (
            self.max_res_area[1] / self.vae_factor
        )
        possible_resolutions: Set[Resolution] = set()

        # --- Generate Landscape-dominant buckets ---
        w = self.min_size
        while (w / self.vae_factor) * (
            self.min_size / self.vae_factor
        ) <= max_tokens and w <= self.max_dim_limit:
            h = self.min_size
            while (w / self.vae_factor) * (
                (h + self.divisible) / self.vae_factor
            ) <= max_tokens and (h + self.divisible) <= self.max_dim_limit:
                h += self.divisible
            if h >= self.min_size:
                possible_resolutions.add((w, h))
            w += self.divisible

        # --- Generate Portrait-dominant buckets ---
        h = self.min_size
        while (self.min_size / self.vae_factor) * (
            h / self.vae_factor
        ) <= max_tokens and h <= self.max_dim_limit:
            w = self.min_size
            while ((w + self.divisible) / self.vae_factor) * (
                h / self.vae_factor
            ) <= max_tokens and (w + self.divisible) <= self.max_dim_limit:
                w += self.divisible
            if w >= self.min_size:
                possible_resolutions.add((w, h))
            h += self.divisible

        # Ensure base resolution is explicitly included if valid
        if (
            (self.base_res[0] / self.vae_factor) * (self.base_res[1] / self.vae_factor)
            <= max_tokens
            and self.base_res[0] >= self.min_size
            and self.base_res[1] >= self.min_size
            and self.base_res[0] <= self.max_dim_limit
            and self.base_res[1] <= self.max_dim_limit
            and self.base_res[0] % self.divisible == 0
            and self.base_res[1] % self.divisible == 0
        ):
            possible_resolutions.add(self.base_res)

        if not possible_resolutions:
            raise ValueError(
                "No valid bucket resolutions generated! Check constraints."
            )

        # NovelAI sorting: width primary, height tie-breaker
        sorted_res = sorted(
            list(possible_resolutions),
            key=lambda r: (r[0] * self.max_dim_limit - r[1]),
        )
        log_aspect_ratios = [math.log(w / h) if h > 0 else 0.0 for w, h in sorted_res]
        return sorted_res, log_aspect_ratios

    def assign_bucket(
        self, width: int, height: int
    ) -> Optional[Tuple[int, int, int, float]]:
        """Maps an image to the closest bucket or prunes extreme ratios."""
        if min(width, height) < self.min_resolution:
            return None

        aspect_ratio = float(width) / max(1.0, float(height))
        log_img_ar = math.log(aspect_ratio)

        best_idx = -1
        min_diff = float("inf")

        for idx, log_bucket_ar in enumerate(self.log_aspect_ratios):
            diff = abs(log_bucket_ar - log_img_ar)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx

        if min_diff > self.max_log_ar_diff or best_idx == -1:
            return None

        target_w, target_h = self.bucket_resolutions[best_idx]
        return best_idx, target_w, target_h, aspect_ratio

    def get_bucket_info(
        self,
        vae_downsample_factor: int = 8,
        latent_channels: int = 32,
    ) -> List[Dict[str, Any]]:
        """Exports bucket metadata schema required by asuka-fm."""
        info = []
        for i, (w, h) in enumerate(self.bucket_resolutions):
            info.append(
                {
                    "bucket_idx": i,
                    "resolution": [w, h],
                    "latents_resolution": [
                        h // vae_downsample_factor,
                        w // vae_downsample_factor,
                    ],
                    "latent_channels": latent_channels,
                    "log_aspect_ratio": round(math.log(w / max(1, h)), 4),
                }
            )
        return info
