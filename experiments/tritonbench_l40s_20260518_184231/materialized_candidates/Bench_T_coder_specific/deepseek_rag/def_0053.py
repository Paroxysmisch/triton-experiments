import torch
import triton
from triton import language as tl
from fast_llm.functional.config import TritonConfig
from fast_llm.engine.config_utils.data_type import DataType

@triton.jit
def triton_mul_relu_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    # Calculate the start of the block
    block_start = tl.program_id(axis=0).to(tl.int64) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel
    input_ = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    mul_result = input_ * other
    tl.store(out_ptr + offsets, tl.maximum(mul_result, tl.zeros_like(mul_result)), mask=mask)

def triton_mul_relu(
    input_,
    other,
    inplace: bool = False,
    out: torch.Tensor | None = None,
):
    """
    A faster triton implementation of element-wise multiplication followed by ReLU.
    """
    if not TritonConfig.TRITON_ENABLED:
        return torch.relu(input_.mul(other))
    assert input_.is_contiguous()
    numel = input_.numel()
    if out is None:
        if inplace:
            out = input_
        else:
            out = torch.empty_like(input_)
    else:
        assert out.is_contiguous()
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    triton_mul_relu_kernel[grid](input_, other, out, numel, block_size=TritonConfig.POINTWISE_BLOCK_SIZE)
    return out
