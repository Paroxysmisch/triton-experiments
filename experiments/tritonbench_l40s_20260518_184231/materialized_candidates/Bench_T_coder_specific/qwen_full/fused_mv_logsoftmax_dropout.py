import torch
import triton
import triton.language as tl
from torch import Tensor
from .triton_utils import get_kernel_meta

@triton.jit
def _fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, out_ptr, n_elements, n_cols, dropout_p, block_size,
    ONE_MINEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr, IS_RCP: tl.constexpr,
    CEIL_NROWS: tl.constexpr, CEIL_NCOLS: tl.constexpr, OUT_SOFTMAX: tl.constexpr,
    DETERMINISTIC: tl.constexpr, RETURN_LOG: tl.constexpr, HAS_MASK: tl.constexpr,
    MASK_VALUE: tl.constexpr, DIM: tl.constexpr, CEIL_N_ELEMENTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_ptr += pid * n_cols
    vec_ptr += pid
    out_ptr += pid * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    vec = tl.load(vec_ptr, eviction_policy='evict_last')
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    max_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')

    for _ in range(0, n_elements, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        a = tl.load(input_ptr + cols, mask=mask, other=0.0)
        max_value = tl.maximum(max_value, a)

        acc += a.to(tl.float32)
        input_ptr += BLOCK_SIZE

    if IS_RCP:
        scale = tl.rcp(tl.sum(tl.exp(acc - max_value), axis=0))
    else:
        scale = 1.0 / (tl.sum(tl.exp(acc - max_value), axis=0) + ONE_MINEPS)

    acc = tl.exp(acc - max_value) * scale

    if CEIL_NROWS == 1:
        acc = tl.where(mask, acc, 0.0)
    elif CEIL_NCOLS == 1:
        acc = tl.where(mask, acc, MASK_VALUE)
    else:
        acc = tl.where(mask, acc, MASK_VALUE)

    if RETURN_LOG:
        acc = max_value + tl.log(acc)
        if OUT_SOFTMAX:
            acc = tl.where(mask, acc, MASK_VALUE)
        elif CEIL_NROWS == 1:
            acc = tl.where(mask, acc, 0.0)
    else:
        acc = tl.where(mask, acc, MASK_VALUE)

    if HAS_MASK:
        vec = tl.where(mask, vec, MASK_VALUE)

    if dropout_p > 0.0:
        keep_prob = 1.0 - dropout_p
        if DETERMINISTIC:
            acc = tl.where(tl.rand(tl.zeros((BLOCK_SIZE,)), dtype=tl.float32) < keep_prob, acc / keep_prob, 0.0)
        else:
            acc = tl.where(tl.rand(tl.zeros((BLOCK_SIZE,)), dtype=tl.float32) < keep_prob, acc / keep_prob, MASK_VALUE)

    acc = acc.to(tl.float16)

    if DIM == 0:
        tl.store(out_ptr + cols, acc, mask=mask)
    elif DIM == 1:
        tl.store(out_ptr + cols * n_cols, acc, mask=mask)

def fused_mv_logsoftmax_dropout(
    input: Tensor, vec: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, dim: int = 0, *, out: Tensor = None
) -> Tensor:
    assert input.is_contiguous()
    assert vec.is_contiguous()
    assert input.dim() >= 2
    assert vec.dim() == 1
    assert 0 <= dim < input.dim()

    input_arg = input
    if dim != input.dim() - 1:
        input_shape = list(input.shape)
        vec_shape = list(vec.shape)
        input = input.transpose(dim, input.dim() - 1).reshape(-1, vec.shape[0])
        vec = vec.reshape(-1)
        assert input.dim() == 2
        assert vec.dim() == 1

    n_rows, n_cols = input.shape
    ceil_nrows = triton.next_power_of_2(n_rows)
    ceil_ncols = triton.next_power_of_2(n_cols)
    n_elements = n_rows * n_cols

    if inplace:
        assert out is None
        out = input_arg
        assert out.stride(-1) == 1
    else:
        assert out is None or out is None
        out = torch.empty_like(input, dtype=input.dtype, memory_format=torch.contiguous)
        assert out.stride(-1) == 1

    ceil_n_elements = triton.next_power_of_2(n_elements)

    use_fp8 = input.dtype == torch.float16 or input.dtype == torch.bfloat16

    _fused_mv_logsoftmax_dropout_kernel[(n_elements,)](
        input,
        vec,
        out,
        n_elements,
        n_cols,
        p,
        128,
        BLOCK_SIZE=n_cols,
        num_warps=4,
        num_stages=2,
        OUT_SOFTMAX=False,
        RETURN_LOG=True,
        DIM=dim,
        CEIL_NROWS=ceil_nrows,
        CEIL_NCOLS=ceil_ncols,
        CEIL_N_ELEMENTS=ceil_n_elements,
        HAS_MASK=False,
        MASK_VALUE=0.0,
        DETERMINISTIC=not training,
        IS_RCP=False,
        ONE_MINEPS=1.0 - torch.finfo(input.dtype).eps if use_fp8 else 0.0,
    )

    if dim != input_arg.dim() - 1:
        out = out.reshape(input_arg.shape).transpose(dim, input_arg.dim() - 1)

    return out
