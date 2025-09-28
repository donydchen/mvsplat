from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerCustomPassthroughCfg:
    name: Literal["custom_passthrough"]
    num_context_views: int
    num_target_views: int


class ViewSamplerCustomPassthrough(ViewSampler[ViewSamplerCustomPassthroughCfg]):
    """
    Pass-through view sampler for custom dataset that handles view sampling internally.
    This sampler does nothing - it relies on the custom dataset to provide pre-sampled views.
    """
    
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        # Custom dataset pre-samples views, so we return identity indices
        # The custom dataset wrapper will handle the actual sampling
        v, _, _ = extrinsics.shape
        context_indices = torch.arange(v, device=device)
        target_indices = torch.arange(v, device=device)
        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views