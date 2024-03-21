from jaxtyping import Float, Int64, Shaped
from torch import Tensor, nn

from ....misc.discrete_probability_distribution import (
    gather_discrete_topk,
    sample_discrete_distribution,
)


class Sampler(nn.Module):
    def forward(
        self,
        probabilities: Float[Tensor, "*batch bucket"],
        num_samples: int,
        deterministic: bool,
    ) -> tuple[
        Int64[Tensor, "*batch 1"],  # index
        Float[Tensor, "*batch 1"],  # probability density
    ]:
        return (
            gather_discrete_topk(probabilities, num_samples)
            if deterministic
            else sample_discrete_distribution(probabilities, num_samples)
        )

    def gather(
        self,
        index: Int64[Tensor, "*batch sample"],
        target: Shaped[Tensor, "..."],  # *batch bucket *shape
    ) -> Shaped[Tensor, "..."]:  # *batch sample *shape
        """Gather from the target according to the specified index. Handle the
        broadcasting needed for the gather to work. See the comments for the actual
        expected input/output shapes since jaxtyping doesn't support multiple variadic
        lengths in annotations.
        """
        bucket_dim = index.ndim - 1
        while len(index.shape) < len(target.shape):
            index = index[..., None]
        broadcasted_index_shape = list(target.shape)
        broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim]
        index = index.broadcast_to(broadcasted_index_shape)
        return target.gather(dim=bucket_dim, index=index)
