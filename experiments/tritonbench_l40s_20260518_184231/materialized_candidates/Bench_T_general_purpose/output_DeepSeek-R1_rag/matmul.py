import triton
import triton.language as tl
import torch

@triton.jit
def dot_product_kernel(
    x_ptr, y_ptr, out_ptr,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    product = x * y
    result = tl.sum(product, axis=0)

    if pid == 0:
        tl.store(out_ptr, result)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(input, other, *, out=None):
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise RuntimeError("matmul: out argument is not supported for 1D inputs")
        if input.size(0) != other.size(0):
            raise ValueError("matmul: input and other must have the same size for 1D case")
        N = input.size(0)
        BLOCK_SIZE = 1024
        N_EXTENDED = triton.next_power_of_2(N)
        out_tensor = torch.empty((), dtype=input.dtype, device=input.device)
        grid = (triton.cdiv(N_EXTENDED, BLOCK_SIZE),)
        dot_product_kernel[grid](input, other, out_tensor, N, BLOCK_SIZE)
        return out_tensor
    elif input.dim() == 2 and other.dim() == 2:
        M, K = input.shape
        K_, N = other.shape
        if K != K_:
            raise ValueError(f"matmul: Incompatible shapes {input.shape} and {other.shape}")
        output = torch.empty((M, N), dtype=input.dtype, device=input.device)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        matmul_kernel[grid](
            input, other, output,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        if out is not None:
            out.copy_(output)
            return out
        return output
    elif input.dim() == 1 and other.dim() == 2:
        K = input.size(0)
        K_, N = other.shape
        if K != K_:
            raise ValueError(f"matmul: Incompatible shapes {input.shape} and {other.shape}")
        output = torch.empty((N,), dtype=input.dtype, device=input.device)
        input_2d = input.unsqueeze(0)
        output_2d = torch.empty((1, N), dtype=input.dtype, device=input.device)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
        grid = (1, triton.cdiv(N, BLOCK_SIZE_N))
        matmul_kernel[grid](
            input_2d, other, output_2d,
            1, N, K,
            input_2d.stride(0), input_2d.stride(1),
            other.stride(0), other.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        output.copy_(output_2d.squeeze(0))
        if out is not None:
            out.copy_(output)
            return out
        return output
    elif input.dim() == 2 and other.dim() == 1:
        M, K = input.shape
        K_ = other.size(0)
        if K != K_:
            raise ValueError(f"matmul: Incompatible shapes {input.shape} and {other.shape}")
        output = torch.empty((M,), dtype=input.dtype, device=input.device)
        other_2d = other.unsqueeze(1)
        output_2d = torch.empty((M, 1), dtype=input.dtype, device=input.device)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
        grid = (triton.cdiv(M, BLOCK_SIZE_M), 1)
        matmul_kernel[grid](
            input, other_2d, output_2d,
            M, 1, K,
            input.stride(0), input.stride(1),
            other_2d.stride(0), other_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        output.copy_(output_2d.squeeze(1))
        if out is not None:
            out.copy_(output)
            return out
        return output
    else:
        raise NotImplementedError("Batched matmul is not implemented in this example")
