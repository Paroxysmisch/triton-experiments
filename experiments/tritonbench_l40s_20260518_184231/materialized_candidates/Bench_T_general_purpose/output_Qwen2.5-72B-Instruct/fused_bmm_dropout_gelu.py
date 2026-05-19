import triton
import triton.language as tl

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    X,  # Pointer to the first input tensor
    Y,  # Pointer to the second input tensor
    Z,  # Pointer to the output tensor
    B,  # Batch size
    N,  # Dimension N
    M,  # Dimension M
    P,  # Dimension P
    p,  # Dropout probability
    training,  # Training flag
    approximate,  # Approximation method for GELU
    seed,  # Seed for dropout
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % N
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % P
    offs_k = tl.arange(0, BLOCK_SIZE_P)
    X = X + (offs_am[:, None] * M + offs_k[None, :])
    Y = Y + (offs_k[:, None] * P + offs_bn[None, :])
    Z = Z + (offs_am[:, None] * P + offs_bn[None, :])

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, M, BLOCK_SIZE_P):
        x = tl.load(X)
        y = tl.load(Y)
        acc += tl.dot(x, y)
        X += BLOCK_SIZE_P
        Y += BLOCK_SIZE_P * P

    # Apply dropout
    if training:
        rng = tl.rand(seed, (pid_m, pid_n))
        mask = rng > p
        acc = tl.where(mask, acc / (1 - p), 0.0)

    # Apply GELU
    if approximate == 'none':
        acc = 0.5 * acc * (1 + tl.tanh(tl.sqrt(2 / 3.141592653589793) * (acc + 0.044715 * acc * acc * acc)))
    elif approximate == 'tanh':
        acc = acc * tl.sigmoid(1.702 * acc)

    tl.store(Z, acc)

import torch
import triton
import triton.language as tl

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    # Ensure input tensors are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()

    # Generate a seed for dropout
    seed = torch.randint(0, 2**32, (1,), device=input1.device).item()

    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_M']) * triton.cdiv(P, META['BLOCK_SIZE_N']),
    )
    fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out, B, N, M, P, p, training, approximate, seed,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_P=16, GROUP_SIZE_M=8
    )

    return out
