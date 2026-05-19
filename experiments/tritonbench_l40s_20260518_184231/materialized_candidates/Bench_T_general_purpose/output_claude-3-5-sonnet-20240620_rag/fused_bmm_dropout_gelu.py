import torch
import triton
import triton.language as tl

@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))

@triton.jit
def gelu(x, approximate: tl.constexpr):
    if approximate == "none":
        return x * 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))
    elif approximate == "tanh":
        return x * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * tl.pow(x, 3.0))))
    else:
        raise ValueError("approximate must be one of ['none', 'tanh']")


@triton.jit
def fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,  # Dimensions
    drop_p, seed, training, approximate,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    n_mask = offs_n < N
    p_mask = offs_p < P

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for b in range(B):
        for m in range(M):
            input1 = tl.load(input1_ptr + b * N * M + offs_n * M + m, mask=n_mask)
            input2 = tl.load(input2_ptr + b * M * P + m * P + offs_p, mask=p_mask)
            acc += input1[:, None] * input2[None, :]

    output = acc
    if training:
        output = apply_dropout(output, drop_p, seed, pid_n * BLOCK_SIZE_P * N + offs_n[:, None] * BLOCK_SIZE_P + offs_p[None, :])
    output = gelu(output, approximate)

    tl.store(output_ptr + pid_n * BLOCK_SIZE_N * P + offs_n[:, None] * P + offs_p[None, :], output, mask=n_mask[:, None] & p_mask[None, :])


def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
    elif inplace:
        out = input1

    grid = (triton.cdiv(N, 32), triton.cdiv(P, 32))
    fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        p, 42, training, approximate,  # seed is fixed for now
        BLOCK_SIZE_N=32, BLOCK_SIZE_P=32
    )
    return out


# Example usage
input1 = torch.randn(2, 32, 64, device='cuda')
input2 = torch.randn(2, 64, 128, device='cuda')
output = fused_bmm_dropout_gelu(input1, input2, training=True, approximate='tanh')
print(output.shape)
