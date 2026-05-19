import triton
import torch
import numpy as np
import triton.language as tl

@triton.autotune(
    configs=[
        {'HAS_SCALE': False, 'WARP': 1},
        {'HAS_SCALE': False, 'WARP': 2},
        {'HAS_SCALE': False, 'WARP': 4},
        {'HAS_SCALE': False, 'WARP': 8},
        {'HAS_SCALE': False, 'WARP': 16},
        {'HAS_SCALE': True, 'WARP': 1},
        {'HAS_SCALE': True, 'WARP': 2},
        {'HAS_SCALE': True, 'WARP': 4},
        {'HAS_SCALE': True, 'WARP': 8},
        {'HAS_SCALE': True, 'WARP': 16},
    ],
    )
@triton.jit
def logsumexp_fwd_kernel(
    X_p,
    Z_p,
    BLOCK_X,
    BLOCK_Z,
    D,
    HAS_SCALE,
    **meta
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    i_n = pid_n * BLOCK_X + tl.arange(0, BLOCK_X)
    i_d = pid_d
    o_d = i_d * BLOCK_Z + tl.arange(0, BLOCK_Z) 
    m_d = o_d // D 
    in_d = o_d % D 
    b_x = tl.load(X_p + i_n)[in_d]

    if HAS_SCALE:
        scale = tl.load(X_p + i_n + D * BLOCK_X)
        b_x = b_x * scale

    b_m = tl.max(b_x, axis=0)
    z = tl.log(tl.sum(tl.exp(b_x - b_m), axis=0)) + b_m
    tl.store(Z_p + o_d, z)

def logsumexp_fwd(x, block_x, block_z, has_scale=False):
    D = block_z
    N = x.shape[0]
    B = block_x
    x_p = tl.pointer(x, tl.mem.global_)
    ND = N * D
    z = torch.zeros((N, D), dtype=x.dtype, device=x.device)
    z_p = tl.pointer(z, tl.mem.global_)
    grid = (tl.next_power_of_2(ND) // BLOCK_Z, N // BLOCK_X)
    logsumexp_fwd_kernel[grid](x_p, z_p, BLOCK_X, BLOCK_Z, D, has_scale,
                               force_nc_cache=True)
    z = z.view(N, -1)
    return z if x.dtype == torch.float32 else z.to(x.dtype)
