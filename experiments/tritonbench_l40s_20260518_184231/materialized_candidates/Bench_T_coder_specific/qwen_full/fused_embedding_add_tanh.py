import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

def embedding_add_tanh_embedding(
    input_indices: Tensor,
    weight: Tensor,
    other: Tensor,
    padding_idx: int = -1,
    max_norm: float = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    out: Tensor = None,
) -> Tensor:
    T = input_indices
    Embedding = weight
    W = other

    V, D = Embedding.shape
    _, S = T.shape

    if out == None:
        S_t = T.reshape((1, -1))
        grad_output = torch.empty((1, S), dtype=Embedding.dtype, device=Embedding.device)
        Y = torch.empty((1, S), dtype=Embedding.dtype, device=Embedding.device)
    else:
        Y = out
        S_t = Y.reshape((1, -1))
        grad_output = torch.ones_like(S_t, dtype=Embedding.dtype, device=Embedding.device)

    if scale_grad_by_freq:
        freq = torch.empty((1, S), dtype=Embedding.dtype, device=Embedding.device)
        EmbeddingGrad = torch.empty_like(Embedding)
    else:
        EmbeddingGrad = torch.empty_like(Embedding)

    if max_norm:
        Norm = torch.empty((1, S), dtype=Embedding.dtype, device=Embedding.device)

    if sparse:
        EmbeddingGrad = EmbeddingGrad.sparse_mask(Embedding)

    for s in range(S):
        i = T[0][s].item()

        if i == padding_idx:
            continue

        d = Embedding[i]

        if max_norm:
            norm = d.norm(norm_type)

            if norm > max_norm:
                d *= max_norm / norm
                d = d.to(Embedding.dtype)

        S = d + W[0][s]
        y = S.tanh()

        if scale_grad_by_freq:
            freq[0][s] = d.numel() / Embedding.numel()
            grad_output[0][s] /= freq[0][s]

        torch.mul_(grad_output[0][s], y)
        d += grad_output[0][s]

        if scale_grad_by_freq:
            d *= freq[0][s]

        if max_norm and norm > max_norm:
            d *= max_norm / norm

        EmbeddingGrad[i] += d

    if padding_idx >= 0:
        EmbeddingGrad[padding_idx] = 0

    if out == None:
        return Y.reshape(T.shape)
    else:
        return grad_output.reshape(T.shape), EmbeddingGrad

class EmbeddingAddTanhFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_indices: Tensor,
        weight: Tensor,
        other: Tensor,
        padding_idx: int = -1,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        out: Tensor = None,
    ) -> Tensor:
        return embedding_add_tanh_embedding(
            input_indices,
            weight,
            other,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            out,
        )

def fused_embedding_add_tanh(
    input_indices: Tensor,
    weight: Tensor,
    other: Tensor,
    *,
    padding_idx: int = -1,
    max_norm: float = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    out: Tensor = None,
) -> Tensor:
    return EmbeddingAddTanhFunction.apply(
        input_indices,
        weight,
        other,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
        out,
    )
