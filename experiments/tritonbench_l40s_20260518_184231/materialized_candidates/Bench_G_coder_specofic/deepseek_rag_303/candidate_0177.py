import torch
import triton
import triton.language as tl
from modelscope.utils.import_utils import smart_open
import warnings

@triton.autotune(
    configs=[
        triton.Config({"BT": 32, "num_warps": 1}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 32, "num_warps": 2}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 32, "num_warps": 4}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 64, "num_warps": 1}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 64, "num_warps": 2}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 64, "num_warps": 4}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 128, "num_warps": 1}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 128, "num_warps": 2}, pre_hook=init_to_zero(["z"])),
        triton.Config({"BT": 128, "num_warps": 4}, pre_hook=init_to_zero(["z"])),
    ],
    key=["S"],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s, z, stride_0_s, stride_1_s, stride_2_s, stride_3_s, stride_0_z, stride_1_z, stride_2_z, stride_3_z,
    B, H, T, S, BT: tl.constexpr, BS: tl.constexpr, num_warps: tl.constexpr,
):
    # Data loading
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)
    # create lower triangular mask
    m_s = pid_t * BT + tl.arange(0, BT) < tl.minimum(T, pid_t * BT + BT)
    p_s = tl.make_block_ptr(
        s + pid_0 * stride_0_s + pid_1 * stride_1_s,
        (B, H, T, S),
        (stride_0_s, stride_1_s, stride_2_s, stride_3_s),
        (pid_b * BT * BS, pid_h * BS, pid_t * BT, 0),
        (BT, BS, 1, 0),
        (BT * BS, 1, 0, S),
    )
    b_s = tl.load(p_s, boundary_check=(0, 1, 2, 3))
    b_c = tl.dot(m_s[:, None] * m_s, b_s)
    # storing b_c result back to z
    p_z = tl.make_block_ptr(
        z + pid_0 * stride_0_z + pid_1 * stride_1_z,
        (B, H, T, S),
        (stride_0_z, stride_1_z, stride_2_z, stride_3_z),
        (pid_b * BT * BS, pid_h * BS, pid_t * BT, 0),
        (BT, BS, 1, 0),
        (BT * BS, 1, 0, S),
    )
    tl.store(p_z, b_c.to(z.dtype.element_ty), boundary_check=(0, 1, 3))


def chunk_global_cumsum_vector(s: torch.Tensor, BT: int = 32, BS: int = 128):
    B, H, T, S = s.shape
    z = torch.empty((B, H, T, S), dtype=s.dtype, device=s.device)
    NT = triton.cdiv(T + BT - 1, BT)
    grid = lambda _: (B, H, NT)
    chunk_global_cumsum_vector_kernel[grid](s, z, *s.stride(), *z.stride(), B, H, T, S, BT=BT, BS=BS, num_warps=1)
    return z
