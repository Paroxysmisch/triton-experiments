import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import FunctionCtx
from ..utils import get_triton_block_size
from .embedding import embedding_forward, embedding_backward


class FusedEmbeddingAddTanhForAutograd(FunctionCtx):
    @staticmethod
    def forward(
        ctx,
        input_indices: Tensor,
        weight: Tensor,
        other: Tensor,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        out: Tensor = None,
    ) -> Tensor:
        ctx.save_for_backward(input_indices, weight)
        ctx.input_indices_shape = input_indices.shape
        ctx.weight_dtype = weight.dtype
        ctx.weight_requires_grad = weight.requires_grad
        ctx.other_strides = other.stride()
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse

        return embedding_forward(
            input_indices,
            weight,
            other,
            padding_idx,
            max_norm,
            norm_type,
            out,
            fused_tanh=True,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        input_indices, weight = ctx.saved_tensors
        input_indices = input_indices.reshape(ctx.input_indices_shape)
        grad_input_indices = torch.ones_like(input_indices, dtype=torch.float32)
        assert not (
            ctx.scale_grad_by_freq
        ), "scale_grad_by_freq is not supported in the fused op yet."
        assert not ctx.sparse, "Sparse case is not supported in the fused op yet."

        grad_weight = embedding_backward(
            grad_input_indices,
            grad_output,
            input_indices,
            padding_idx=-1,
            scale_grad_by_freq=False,
            sparse=False,
        )
        grad_other = torch.tensordot(
            grad_output, weight.to(grad_output.dtype), dims=([1], [0])
        ).reshape(ctx.other_strides)

        if ctx.weight_dtype != grad_weight.dtype:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        if ctx.weight_requires_grad:
            return grad_input_indices, grad_weight, grad_other
        else:
            return grad_input_indices, None, grad_other


def fused_embedding_add_tanh(
    input_indices: Tensor,
    weight: Tensor,
    other: Tensor,
    *,
    padding_idx: int = None,
    max_norm: float = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    out: Tensor = None,
) -> Tensor:
    r"""retrieves the embeddings from the :attr:`weight` matix using
    the indices in :attr:`input_indices`, add the :attr:`other` tensor to it,
    and apply tanh activation.

    Args:
        input_indices (LongTensor): A tensor contains the indices of the embedded
            vectors, it can be of arbitrary shape.
        weight (Tensor): The embedding matrix of shap (`num_embeddings`, `embedding_dim`),
            this argument expects a 2D tensor.
        other (Tensor): The tensor to add to the retrieved embeddings, it must be
            broadcastable with the reshaped input indices (arbitrary shape + `embedding_dim`)
            to a 2D tensor.
        padding_idx (Optional[int]): If specified, the output for this index position
            will be zeroed out and will not contribute to the gradient. Default: `None`.
        max_norm (Optional[float]): If given, each embedding vector with norm larger than
            `max_norm` is renormalized to have norm `max_norm`. Default: `None`.
        norm_type (Optional[float]): The p-norm to compute for the `max_norm` option.
            Default: `2.0`.
        scale_grad_by_freq (Optional[bool]): If `True`, scale gradients by the inverse
            of frequency of the words in the batch. Default: `False`.
        sparse (Optional[bool]): Not supported in this fused op, if `True`, gradient w.r.t.
            `weight` will be a sparse tensor. Default: `False`.
        out (Optional[Tensor]): The output tensor. If ``None`` the output tensor is constructed
            implicitly and returned.

    Returns:
        Tensor: The result of the fused embedding, addition, and tanh operation.
            It's shape is the shape of the input indices tensor plus the embedding dim.

    Example::


        >>> weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        >>> input_indices = torch.tensor([[0, 1, 0]], dtype=torch.long)
        >>> other = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        >>> out = fused_embedding_add_tanh(input_indices, weight, other)
        >>> print(out)
            tensor([[[ 2.0000,  3.0000,  4.0000],
                     [ 5.0000,  6.0000,  7.0000],
                     [ 2.0000,  3.0000,  4.0000]]])
        >>> input_indices = torch.tensor([0, 1, 0]).unsqueeze(0)
        """
    return FusedEmbeddingAddTanhForAutograd.apply(
        input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, out
    )
