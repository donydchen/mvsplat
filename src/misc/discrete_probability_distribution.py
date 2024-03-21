import torch
from einops import reduce
from jaxtyping import Float, Int64
from torch import Tensor


def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    samples = torch.rand((*batch, num_samples), device=pdf.device)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)
