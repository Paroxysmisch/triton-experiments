import triton
import torch

@triton.jit
def dequantize_kernel(b_ptr, b_scale_ptr, fpb_ptr, K, N, stride_b, stride_scale, stride_fpb):
    row = triton.program.serial(blockIdx.x)
    col = triton.program.serial(blockIdx.y)

    if row < K and col < N:
        b_val = triton.program.load(b_ptr + row * stride_b + col)
        b_scale_val = triton.program.load(b_scale_ptr + col)
        fpb_val = b_val * b_scale_val
        triton.program.store(fpb_ptr + row * stride_fpb + col, fpb_val)

def matmul_dequantize_int8(a, b, b_scale, c):
    assert a.is_cuda and b.is_cuda and b_scale.is_cuda and c.is_cuda
    assert a.is_contiguous() and b.is_contiguous() and b_scale.is_contiguous() and c.is_contiguous()

    K, M = a.shape
    K, N = b.shape

    assert M == b_scale.numel()

    b_ptr = triton.pointers.register_buffer(b)
    b_scale_ptr = triton.pointers.register_buffer(b_scale)
    fpb = torch.empty_like(b, device='cuda')
    fpb_ptr = triton.pointers.register_buffer(fpb)

    stride_b = b.stride(0)
    stride_scale = b_scale.stride(0)
    stride_fpb = fpb.stride(0)

    grid = (K, N)
    dequantize_kernel[grid](b_ptr, b_scale_ptr, fpb_ptr, K, N, stride_b, stride_scale, stride_fpb)

    c.copy_(a @ fpb)
