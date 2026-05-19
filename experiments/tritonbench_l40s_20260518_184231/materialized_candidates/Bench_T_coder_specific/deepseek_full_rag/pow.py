import torch
import triton
import triton.language as tl

@triton.jit
def pow_kernel(x_ptr, exponent, out_ptr, n_elements,
               X_BLOCK_SIZE: tl.constexpr,
               EXPONENT_IS_TENSOR: tl.constexpr,
               EXPONENT_TENSOR_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * X_BLOCK_SIZE
    offsets = block_start + tl.arange(0, X_BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x ** exponent
    if EXPONENT_IS_TENSOR:
        exponent_ptr = exponent.to_tensor_ptr()
        exponent_block_start = pid * EXPONENT_TENSOR_BLOCK_SIZE
        exponent_offsets = exponent_block_start + tl.arange(0, EXPONENT_TENSOR_BLOCK_SIZE)
        exponent_mask = exponent_offsets < n_elements
        exponent_val = tl.load(exponent_ptr + exponent_offsets, mask=exponent_mask).to(tl.float32)
        result = x ** exponent_val
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input: torch.Tensor, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    if isinstance(exponent, torch.Tensor):
        exponent_is_tensor = True
        exponent_tensor_broadcasted = broadcast_shapes(input.shape, exponent.shape)
        exponent_tensor_size = math.prod(exponent_tensor_broadcasted.shape)
        exponent_tensor_block_size = get_block_size(exponent_tensor_size)
        exponent = triton.reinterpret(exponent, dtype=exponent_tensor_broadcasted.dtype)
    else:
        exponent_is_tensor = False
        exponent_tensor_block_size = 0
    input_size = math.prod(input.shape)
    input_block_size = get_block_size(input_size)
    grid = lambda meta: (triton.cdiv(input_size, meta['X_BLOCK_SIZE']),)
    pow_kernel[grid](input, exponent, out, input_size,
                     EXPONENT_IS_TENSOR=exponent_is_tensor,
                     EXPONENT_TENSOR_BLOCK_SIZE=exponent_tensor_block_size)
    return out
