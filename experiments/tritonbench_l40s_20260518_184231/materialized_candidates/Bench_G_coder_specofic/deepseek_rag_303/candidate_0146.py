import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.autograd.function import requires_grad
from torchvision.ops.misc import _contiguous_cache
from typing import Optional

from torch.cuda.amp import custom_fwd

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr: tl.tensor,
    b_ptr: tl.tensor,
    out_ptr: tl.tensor,
    w_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    b_add_ptr: tl.tensor,
    ACTIVATION: int,
    MUL_IN: tl.constexpr,
    CLIP_IN: tl.constexpr,
    MUL_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < tl.num_programs(0) * BLOCK_SIZE

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(tl.num_programs(0) * BLOCK_SIZE,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(tl.num_programs(0) * BLOCK_SIZE,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(tl.num_programs(0) * BLOCK_SIZE,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )

    w_block_ptr = tl.make_block_ptr(
        base=w_ptr, shape=(tl.num_programs(0) * BLOCK_SIZE,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )

    scale_block_ptr = tl.make_block_ptr(
        base=scale_ptr, shape=(tl.num_programs(0) * BLOCK_SIZE,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    x = tl.load(x_block_ptr)
    b = tl.load(b_block_ptr)
    w = tl.load(w_block_ptr)
    add = tl.load(b_add_ptr)
    scale = tl.load(scale_block_ptr)
    if CLIP_IN:
        x = tl.where(x > 127, 127, x)
        x = tl.where(x < -127, -127, x)
    x = x.to(tl.float32)

    if MUL_IN:
        x = x * w
    if MUL_OUT:
        b = b * w

    b = b.to(tl.float32)
    scale = scale.to(tl.float32)

    x = tl.math.round(add + scale * (x * w + b))

    if CLIP_IN:
        x = tl.where(x > 127, 127, x)
        x = tl.where(x < -127, -127, x)
    if ACTIVATION == 0:
        pass
    elif ACTIVATION == 1:
        x = tl.math.sigmoid(x)
    elif ACTIVATION == 2:
        x = tl.math.relu(x)
    elif ACTIVATION == 3:
        pass
    elif ACTIVATION == 4:
        x = tl.math.tanh(x)
    elif ACTIVATION == 5:
        pass

    tl.store(out_block_ptr, x, mask=mask)


@requires_grad
def fused_add_mul_activation_torch(
        in_out_tensor: Tensor,
        weight: Tensor,
        bias_add: Tensor,
        scale: Tensor,
        activation_type: Optional[str],
        mul_in: bool = False,
        clip_in: bool = False,
        mul_out: bool = False
) -> Tensor:
    in_out_tensor = in_out_tensor.to(torch.float16)
    assert in_out_tensor.is_contiguous(), "tensor must be contiguous"
    assert weight.is_contiguous(), "weight tensor must be contiguous"
    assert bias_add.is_contiguous(), "bias tensor must be contiguous"
    assert scale.is_contiguous(), "scale tensor must be contiguous"
    assert bias_add.shape[0] == scale.shape[0] == weight.shape[0], f"First dim mismatch: {bias_add.shape[0]}, {scale.shape[0]}, {weight.shape[0]}"
    assert bias_add.shape[0] == in_out_tensor.shape[1], f"{bias_add.shape[0]} != {in_out_tensor.shape[1]}"
    assert in_out_tensor.is_contiguous(), f"{in_out_tensor.stride()}"
    assert in_out_tensor.stride(-1) == 1, "The last dim of `in_out_tensor` should be contiguous"
    assert bias_add.stride(-1) == 1, "The last dim of `bias_add` should be contiguous"
    assert bias_add.stride(-1) == 1, "The last dim of `bias_add` should be contiguous"

    keep_dtype = in_out_tensor.dtype
    in_out_tensor = in_out_tensor.to(torch.float16)
    weight = weight.to(torch.float16)

    if activation_type == "sigmoid":
        activation_type_enum = 1
    elif activation_type == "relu":
        activation_type_enum = 2
    elif activation_type == "swish":
        activation_type_enum = 5
    else:
        assert activation_type is None
        activation_type_enum = 0

    grid = lambda _: (triton.cdiv(in_out_tensor.shape[1], _),)  # noqa: E731

    fused_add_mul_activation_kernel[grid](
        in_out_tensor,
        bias_add,
        in_out_tensor,
        weight,
        scale,
        bias_add,
        activation_type_enum,
        mul_in,
        clip_in,
        mul_out,
        BLOCK_SIZE=512,
        num_warps=1,
    )
    if keep_dtype != torch.float16:
        in_out_tensor = in_out_tensor.to(keep_dtype)

    return in_out_tensor
