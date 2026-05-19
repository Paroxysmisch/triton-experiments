import triton
import triton.language as tl

# Constants
BLOCK_N = 128
BLOCK_K = 32
SPLIT_N = 8
EVEN_K = 1
ADD_INPUTS = 1
CAST_TYPE = tl.float16

# Triton Kernel
@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices, n, k, m, batch_size, grid_size,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_n = pid % (n // BLOCK_N)
    block_k = pid // (n // BLOCK_N)
    row = block_n * BLOCK_N
    col = block_k * BLOCK_K

    x = tl.load(input_ptr + row * k + col, mask=col < k, other=0.0)
    y = tl.zeros((BLOCK_N,), dtype=CAST_TYPE)
    for s in range(SPLIT_N):
        lora_row = tl.load(lora_indices + pid * SPLIT_N + s)
        lora_weight = tl.load(lora_ptr + lora_row * k + col, mask=col < k, other=0.0)
        y += x * lora_weight

    if ADD_INPUTS:
        out = tl.load(out_ptr + row * m + col, mask=col < m, other=0.0)
        out += y
        tl.store(out_ptr + row * m + col, out, mask=col < m)
    else:
        tl.store(out_ptr + row * m + col, y, mask=col < m)

# Triton Wrapper Function
@torch.inference_mode()
def _bgmv_expand_slice(input_ptr, lora_ptr, out_ptr, lora_indices, n, k, m, batch_size):
    # Validate shapes
    assert input_ptr.shape == (batch_size, n, k)
    assert lora_ptr.shape == (batch_size, n, k)
    assert out_ptr.shape == (batch_size, n, m)
    assert lora_indices.shape == (batch_size,)

    # Prepare grid configuration
    grid_size = (n // BLOCK_N) * (k // BLOCK_K)

    # Launch the Triton kernel
    _bgmv_expand_slice_kernel[grid_size, BLOCK_N](input_ptr, lora_ptr, out_ptr, lora_indices, n, k, m, batch_size, grid_size,
                                                BLOCK_N, BLOCK_K, SPLIT_N, EVEN_K, ADD_INPUTS, CAST_TYPE)

    return out_ptr
