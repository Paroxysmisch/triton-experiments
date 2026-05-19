import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s_ptr, o_ptr,
    stride_sz, stride_st, stride_tt,
    T: tl.constexpr, S: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
):
    # Compute indices for the current thread block
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Load block offsets
    t_offs = pid_t * BT + tl.arange(0, BT)
    s_offs = pid_s * BS + tl.arange(0, BS)
    
    # Create mask for upper triangular computation
    m_s = s_offs[:, None] <= t_offs[None, :]
    
    # Load source block
    s_block_ptr = s_ptr + pid_s * stride_sz + pid_t * stride_st
    s = tl.load(s_block_ptr + t_offs * stride_tt, mask=m_s, other=0.0)
    
    # Compute cumulative sum
    o = tl.sum(s, axis=0)
    
    # Store result
    o_block_ptr = o_ptr + pid_t * stride_tt
    tl.store(o_block_ptr + t_offs, o)

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr, h_ptr,
    h0_ptr, ht_ptr,
    stride_kz, stride_kt,
    stride_vz, stride_vt,
    stride_gz, stride_gt,
    stride_hz, stride_ht,
    T: tl.constexpr, Z: tl.constexpr,
    BT: tl.constexpr, BZ: tl.constexpr,
    has_h0: tl.constexpr, store_ht: tl.constexpr,
):
    # Compute indices
    pid_t = tl.program_id(0)
    pid_z = tl.program_id(1)
    
    # Load block offsets
    t_offs = pid_t * BT + tl.arange(0, BT)
    z_offs = pid_z * BZ + tl.arange(0, BZ)
    
    # Initialize masks
    t_mask = t_offs < T
    z_mask = z_offs < Z
    
    # Load blocks
    k_block_ptr = k_ptr + pid_z * stride_kz + pid_t * stride_kt
    v_block_ptr = v_ptr + pid_z * stride_vz + pid_t * stride_vt
    g_block_ptr = g_ptr + pid_z * stride_gz + pid_t * stride_gt
    
    k = tl.load(k_block_ptr + t_offs, mask=t_mask, other=0.0)
    v = tl.load(v_block_ptr + t_offs, mask=t_mask, other=0.0)
    g = tl.load(g_block_ptr + t_offs, mask=t_mask, other=0.0)
    
    # Initialize h with h0 if provided
    h = tl.zeros([BZ], dtype=tl.float32)
    if has_h0:
        h0 = tl.load(h0_ptr + z_offs, mask=z_mask)
        h = h + h0
    
    # Compute gated accumulation
    h = h + tl.sum(k * v * g)
    
    # Store results
    h_block_ptr = h_ptr + pid_z * stride_hz + pid_t * stride_ht
    tl.store(h_block_ptr + z_offs, h, mask=z_mask)
    
    # Store final state if requested
    if store_ht and pid_t == (T // BT - 1):
        tl.store(ht_ptr + z_offs, h, mask=z_mask)

def fwd_pre(g: torch.Tensor) -> torch.Tensor:
    T, B, H = g.shape
    g_cum = g.cumsum(dim=0)
    g_cum = g_cum.reshape(T, B * H)
    return g_cum

def fwd_inner(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor = None,
    store_ht: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, B, H, D = k.shape
    Z = B * H
    
    # Configure block sizes
    BT = 32
    BZ = 32
    
    # Initialize output tensor
    h = torch.zeros((T, Z), device=k.device, dtype=k.dtype)
    ht = torch.zeros((Z,), device=k.device, dtype=k.dtype) if store_ht else None
    
    # Launch kernel
    grid = (triton.cdiv(T, BT), triton.cdiv(Z, BZ))
    chunk_gated_abc_fwd_kernel_h[grid](
        k.reshape(T, Z).contiguous(),
        v.reshape(T, Z).contiguous(),
        g.reshape(T, Z).contiguous(),
        h,
        h0 if h0 is not None else h.new_zeros((0,)),
        ht if ht is not None else h.new_zeros((0,)),
        Z, T,  # strides
        1, 1,  # strides
        1, 1,  # strides
        1, 1,  # strides
        T, Z,
        BT, BZ,
        h0 is not None,
        store_ht,
    )
    
    return h, ht
