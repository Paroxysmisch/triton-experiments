import torch
import triton
import triton.language as tl
from .triton_utils.kernels import silu

@triton.jit
def quant_fused_matmul_248_kernel(
    a_ptr,
    c_ptr,
    b1_ptr,
    scales1_ptr,
    zeros1_ptr,
    g1_ptr,
    b2_ptr,
    scales2_ptr,
    zeros2_ptr,
    g2_ptr,
    M,
    N,
    K,
    bits,
    maxq,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M

    b1_ptrs = b1_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
    b2_ptrs = b2_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
    g1_ptrs = g1_ptr + offs_k
    g2_ptrs = g2_ptr + offs_k

    scales1_ptrs = scales1_ptr + offs_bn[None, :]
    scales2_ptrs = scales2_ptr + offs_bn[None, :]
    zeros1_ptrs = zeros1_ptr + (offs_bn[None, :] // infearure_per_bits)
    zeros2_ptrs = zeros2_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g1_idx = tl.load(g1_ptrs)
        g2_idx = tl.load(g2_ptrs)

        scales1 = tl.load(scales1_ptrs + g1_idx[:, None] * stride_scales)
        scales2 = tl.load(scales2_ptrs + g2_idx[:, None] * stride_scales)

        zeros1 = tl.load(zeros1_ptrs + g1_idx[:, None] * stride_zeros)
        zeros1 = (zeros1 >> zeros_shifter[None, :]) & maxq
        zeros1 = zeros1 + 1

        zeros2 = tl.load(zeros2_ptrs + g2_idx[:, None] * stride_zeros)
        zeros2 = (zeros2 >> zeros_shifter[None, :]) & maxq
        zeros2 = zeros2 + 1

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)

        b1 = (b1 >> shifter[:, None]) & maxq
        b1 = (b1 - zeros1) * scales1
        accumulator1 += tl.dot(a, b1)

        b2 = (b2 >> shifter[:, None]) & maxq
        b2 = (b2 - zeros2) * scales2
        accumulator2 += tl.dot(a, b2)

        a_ptrs += BLOCK_SIZE_K
        b1_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        b2_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g1_ptrs += BLOCK_SIZE_K
        g2_ptrs += BLOCK_SIZE_K

    accumulator1 = silu(accumulator1)
    c = accumulator1 * accumulator2
    c = c.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

class FusedLlamaMLPForQuantizedModel:
    def __init__(self, gate_proj, down_proj, up_proj):
        self.infeatures = gate_proj.infeatures
        self.intermediate_size = gate_proj.outfeatures
        self.outfeatures = down_proj.outfeatures
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def triton_llama_mlp(self, x):
        with torch.cuda.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size,)
            x = x.reshape(-1, x.shape[-1])
            M, K = x.shape
            N = self.intermediate_size
            c = torch.empty((M, N), device=x.device, dtype=torch.float16)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
            quant_fused_matmul_248_kernel[grid](
                x,
                c,
                self.gate_proj.qweight,
                self.gate_proj.scales,
                self.gate_proj.qzeros,
                self.gate_proj.g_idx,
                self.up_proj.qweight,
                self.up_proj.scales,
                self.up_proj.qzeros,
                self.up_proj.g_idx,
                M,
                N,
                K,
                self.bits,
                self.maxq,
                x.stride(0),
                x.stride(1),
                self.gate_proj.qweight.stride(0),
                self.gate_proj.qweight.stride(1),
                c.stride(0),
                c.stride(1),
                self.gate_proj.scales.stride(0),
                self.gate_proj.qzeros.stride(0),
            )
            c = c.reshape(out_shape)
            return c
