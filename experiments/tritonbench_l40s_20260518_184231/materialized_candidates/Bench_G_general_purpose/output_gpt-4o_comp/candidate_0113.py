import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3, primals_1, primals_2,
    out_ptr0, out_ptr1,
    S, D, RBLOCK,
    **meta
):
    pid = tl.program_id(axis=0)
    num_blocks = (D + RBLOCK - 1) // RBLOCK
    start = pid * RBLOCK
    end = min(start + RBLOCK, D)

    # Shared buffers for reduction
    tmp3_mean = tl.zeros([1], dtype=tl.float32)
    tmp3_m2 = tl.zeros([1], dtype=tl.float32)
    tmp3_weight = tl.zeros([1], dtype=tl.float32)

    # Load input and compute mean and variance using Welford's algorithm
    for i in range(start, end):
        x = primals_3[pid, i]
        delta = x - tmp3_mean[0]
        tmp3_weight[0] += 1
        tmp3_mean[0] += delta / tmp3_weight[0]
        delta2 = x - tmp3_mean[0]
        tmp3_m2[0] += delta * delta2

    # Store mean, variance, and count in out_ptr0
    if pid == 0:
        out_ptr0[pid, 0] = tmp3_mean[0]
        out_ptr0[pid, 1] = tmp3_m2[0] / tmp3_weight[0]
        out_ptr0[pid, 2] = tmp3_weight[0]

    # Normalize input
    mean = out_ptr0[pid, 0]
    var = out_ptr0[pid, 1]
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    for i in range(start, end):
        x = primals_3[pid, i]
        x_norm = (x - mean) * inv_std
        if primals_1 is not None and primals_2 is not None:
            x_norm = x_norm * primals_1[i] + primals_2[i]
        out_ptr1[pid, i] = x_norm


import torch

def fused_native_layer_norm(primals_3, primals_1=None, primals_2=None):
    S, D = primals_3.shape
    RBLOCK = 128  # Example block size, can be tuned
    grid = (S,)

    # Prepare output buffers
    buf0 = torch.empty((S, 3), dtype=torch.float32, device=primals_3.device)
    buf3 = torch.empty_like(primals_3)
    buf4 = torch.empty_like(primals_3)

    # Launch Triton kernel
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, primals_1, primals_2,
        buf0, buf4,
        S, D, RBLOCK
    )

    return buf4, buf0, buf3
