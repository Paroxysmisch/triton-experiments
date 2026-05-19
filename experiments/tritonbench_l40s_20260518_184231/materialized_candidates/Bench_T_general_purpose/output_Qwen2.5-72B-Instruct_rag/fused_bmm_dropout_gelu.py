import triton
import triton.language as tl

@triton.jit
def gelu(x, approximate='none'):
    if approximate == 'tanh':
        y = 0.5 * x * (1 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    else:
        y = 0.5 * x * (1 + tl.erf(x / 1.41421356237))
    return y

@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    drop_p, seed, training, approximate,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_p = (P + BLOCK_SIZE_P - 1) // BLOCK_SIZE_P
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks = num_blocks_n * num_blocks_p * num_blocks_m
    block_id = pid % num_blocks
    block_id_n = block_id // (num_blocks_p * num_blocks_m)
    block_id_p = (block_id % (num_blocks_p * num_blocks_m)) // num_blocks_m
    block_id_m = block_id % num_blocks_m

    offsets_n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_p = block_id_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offsets_m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_b = tl.arange(0, B)

    mask_n = offsets_n < N
    mask_p = offsets_p < P
    mask_m = offsets_m < M

    input1 = tl.load(input1_ptr + offsets_b[:, None, None] * N * M + offsets_n[None, :, None] * M + offsets_m[None, None, :],
                     mask=mask_n[None, :, None] & mask_m[None, None, :], other=0.0)
    input2 = tl.load(input2_ptr + offsets_b[:, None, None] * M * P + offsets_m[None, :, None] * P + offsets_p[None, None, :],
                     mask=mask_m[None, :, None] & mask_p[None, None, :], other=0.0)

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for m in range(0, M, BLOCK_SIZE_M):
        a = tl.load(input1_ptr + offsets_b[:, None, None] * N * M + offsets_n[None, :, None] * M + m + offsets_m[None, None, :],
                    mask=mask_n[None, :, None] & mask_m[None, None, :], other=0.0)
        b = tl.load(input2_ptr + offsets_b[:, None, None] * M * P + m + offsets_m[None, :, None] * P + offsets_p[None, None, :],
                    mask=mask_m[None, :, None] & mask_p[None, None, :], other=0.0)
        acc += tl.dot(a, b)

    if training:
        acc = apply_dropout(acc, drop_p, seed, block_id)

    acc = gelu(acc, approximate)

    tl.store(output_ptr + offsets_b[:, None, None] * N * P + offsets_n[None, :, None] * P + offsets_p[None, None, :],
             acc, mask=mask_n[None, :, None] & mask_p[None, None, :])

import torch
import triton
import triton.language as tl

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    assert input1.dim() == 3 and input2.dim() == 3, "Input tensors must be 3D"
    assert input1.shape[0] == input2.shape[0], "Batch sizes must match"
    assert input1.shape[2] == input2.shape[1], "Matrix dimensions must be compatible for batch matrix multiplication"

    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    seed = torch.randint(0, 2**32, (1,), device=input1.device).item()

    grid = lambda meta: (
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        (P + meta['BLOCK_SIZE_P'] - 1) // meta['BLOCK_SIZE_P'],
        (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        B
    )

    fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        p, seed, training, approximate,
        BLOCK_SIZE_B=1, BLOCK_SIZE_N=128, BLOCK_SIZE_M=64, BLOCK_SIZE_P=128
    )

    return out
