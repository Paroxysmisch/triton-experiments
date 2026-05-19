import torch
import triton
import triton.language as tl

@triton.jit
def fused_cosine_embedding_loss_with_normalization_triton(input1, input2, target, margin, reduction):
    dim = 1
    N = input1.shape[0]
    D = input1.shape[1]
    pid = tl.program_id(0)
    offset_n = pid * N
    n_mask = (offset_n + tl.arange(0, N)) < input1.shape[0]
    d_mask = tl.arange(0, D) < D

    input1 = tl.load(input1 + offset_n[:, None] * N + d_mask[None, :] * D, n_mask[:, None] & d_mask[None, :], eviction_policy='evict_last')
    input2 = tl.load(input2 + offset_n[:, None] * N + d_mask[None, :] * D, n_mask[:, None] & d_mask[None, :], eviction_policy='evict_last')
    target = tl.load(target + offset_n, n_mask, eviction_policy='evict_last')

    input1_norm = tl.sum(input1 * input1, axis=1)
    input1_norm = tl.where(input1_norm == 0., 1e-12, input1_norm)
    input1 = input1 / tl.sqrt(input1_norm[:, None])

    input2_norm = tl.sum(input2 * input2, axis=1)
    input2_norm = tl.where(input2_norm == 0., 1e-12, input2_norm)
    input2 = input2 / tl.sqrt(input2_norm[:, None])

    cos_dist = (input1 * input2).sum(axis=1)
    cos_dist = tl.where(target == 1, cos_dist, -cos_dist)
    loss = tl.where(target == 1, -cos_dist, tl.maximum(margin - cos_dist, 0))

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return tl.sum(loss) / tl.sum(n_mask & (target == 1))
    elif reduction == 'sum':
        return tl.sum(loss)
    else:
        raise ValueError(f"reduction can only be 'none', 'mean' or 'sum', but got {reduction}")

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    dim = 1
    assert input1.shape == input2.shape
    assert input1.shape[dim] > 1
    assert input2.shape[dim] > 1
    assert target.shape == input1.shape[0:1]
    assert target.max() <= 1 and target.min() >= -1
    N, C = input1.shape

    input1 = input1 / input1.norm(dim=dim, keepdim=True)
    input2 = input2 / input2.norm(dim=dim, keepdim=True)

    cos_dist = (input1 * input2).sum(dim=dim)
    cos_dist = torch.where(target == 1, cos_dist, -cos_dist)
    loss = torch.where(target == 1, -cos_dist, torch.max(margin - cos_dist, torch.tensor(0.0, device=cos_dist.device)))

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.sum(loss) / torch.sum(target == 1)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError(f"reduction can only be 'none', 'mean' or 'sum', but got {reduction}")
