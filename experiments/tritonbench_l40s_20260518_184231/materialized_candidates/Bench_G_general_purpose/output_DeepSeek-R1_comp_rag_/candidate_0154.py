import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr,
    initial_state_ptr,
    final_state_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_k = k_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_v = v_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_h = h_ptr + i_bh * T * D + i_t * BT * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE and i_t == 0:
        b_h += tl.load(initial_state_ptr + i_bh * D + o_d, mask=mask, other=0.0).to(tl.float32)

    for i in range(BT):
        current_time = i_t * BT + i
        mask_t = mask & (current_time < T)

        k = tl.load(p_k, mask=mask_t, other=0.0)
        v = tl.load(p_v, mask=mask_t, other=0.0)

        # Example decay function: d_b = exp(-exp(k)), adjust as needed
        d_b = tl.exp(-tl.exp(k))
        d_i = 1.0 - d_b

        b_h = b_h * d_b + (k * v) * d_i
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_t)

        p_k += D
        p_v += D
        p_h += D

    if STORE_FINAL_STATE:
        final_h_ptr = final_state_ptr + i_bh * D + o_d
        tl.store(final_h_ptr, b_h.to(final_h_ptr.dtype.element_ty), mask=mask)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, h_ptr, o_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_q = q_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_k = k_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_h = h_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_o = o_ptr + i_bh * T * D + i_t * BT * D + o_d

    b_o = tl.zeros([BD], dtype=tl.float32)

    for i in range(BT):
        current_time = i_t * BT + i
        mask_t = mask & (current_time < T)

        q = tl.load(p_q, mask=mask_t, other=0.0)
        k = tl.load(p_k, mask=mask_t, other=0.0)
        h = tl.load(p_h, mask=mask_t, other=0.0)

        d_i = 1.0 - tl.exp(-tl.exp(k))  # Match forward kernel's d_i calculation

        b_o += q * h * d_i

        p_q += D
        p_k += D
        p_h += D

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    k_ptr, v_ptr, h_ptr, do_ptr, dh_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_k = k_ptr + i_bh * T * D + (i_t * BT + BT - 1) * D + o_d
    p_v = v_ptr + i_bh * T * D + (i_t * BT + BT - 1) * D + o_d
    p_do = do_ptr + i_bh * T * D + (i_t * BT + BT - 1) * D + o_d
    p_dh = dh_ptr + i_bh * T * D + (i_t * BT + BT - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(BT-1, -1, -1):
        current_time = i_t * BT + i
        mask_t = mask & (current_time < T)

        k = tl.load(p_k, mask=mask_t, other=0.0)
        v = tl.load(p_v, mask=mask_t, other=0.0)
        do = tl.load(p_do, mask=mask_t, other=0.0)

        d_b = tl.exp(-tl.exp(k))
        d_i = 1.0 - d_b

        b_dh += do * d_i * v  # Contribution from current step
        dh = b_dh * k  # Assuming derivative through k*v term

        tl.store(p_dh, dh.to(p_dh.dtype.element_ty), mask=mask_t)

        b_dh *= d_b  # Backpropagate through decay

        p_k -= D
        p_v -= D
        p_do -= D
        p_dh -= D

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q_ptr, k_ptr, v_ptr, h_ptr,
    do_ptr, dq_ptr, dk_ptr, dv_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_q = q_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_k = k_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_h = h_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_do = do_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_dq = dq_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_dk = dk_ptr + i_bh * T * D + i_t * BT * D + o_d
    p_dv = dv_ptr + i_bh * T * D + i_t * BT * D + o_d

    for i in range(BT):
        current_time = i_t * BT + i
        mask_t = mask & (current_time < T)

        q = tl.load(p_q, mask=mask_t, other=0.0)
        k = tl.load(p_k, mask=mask_t, other=0.0)
        h = tl.load(p_h, mask=mask_t, other=0.0)
        do = tl.load(p_do, mask=mask_t, other=0.0)

        d_i = 1.0 - tl.exp(-tl.exp(k))
        d_b = tl.exp(-tl.exp(k))

        # Compute gradients
        dq = do * h * d_i
        dh_contrib = do * q * d_i
        dk_di = -tl.exp(-tl.exp(k)) * tl.exp(k) * (q * h - k * v)  # Example derivative
        dk = dh_contrib * v * dk_di
        dv = dh_contrib * k

        tl.store(p_dq, dq.to(p_dq.dtype.element_ty), mask=mask_t)
        tl.store(p_dk, dk.to(p_dk.dtype.element_ty), mask=mask_t)
        tl.store(p_dv, dv.to(p_dv.dtype.element_ty), mask=mask_t)

        p_q += D
        p_k += D
        p_h += D
        p_do += D
        p_dq += D
        p_dk += D
        p_dv += D

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state: Optional[torch.Tensor], output_final_state: bool):
        B, H, T, D = q.shape
        BT, BD = 64, min(64, triton.next_power_of_2(D))
        num_warps = 4 if BD == 64 else 2

        h = torch.empty_like(k, dtype=torch.float32)
        final_state = torch.empty(B, H, D, device=q.device, dtype=torch.float32) if output_final_state else None

        # Launch kernel_h
        grid_h = (triton.cdiv(D, BD), triton.cdiv(T, BT), B * H)
        chunk_retention_fwd_kernel_h[grid_h](
            k, v, h,
            initial_state,
            final_state,
            T, D,
            BT=BT,
            BD=BD,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=num_warps
        )

        # Launch kernel_o
        o = torch.empty_like(q, dtype=torch.float32)
        grid_o = (triton.cdiv(D, BD), triton.cdiv(T, BT), B * H)
        chunk_retention_fwd_kernel_o[grid_o](
            q, k, h, o,
            T, D,
            BT=BT, BD=BD,
            num_warps=num_warps
        )
        o = o.to(q.dtype)

        ctx.save_for_backward(q, k, v, h, initial_state)
        ctx.output_final_state = output_final_state
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state):
        q, k, v, h, initial_state = ctx.saved_tensors
        B, H, T, D = q.shape
        BT, BD = 64, min(64, triton.next_power_of_2(D))
        num_warps = 4 if BD == 64 else 2

        dh = torch.empty_like(h, dtype=torch.float32)
        grid_dh = (triton.cdiv(D, BD), triton.cdiv(T, BT), B * H)
        chunk_retention_bwd_kernel_dh[grid_dh](
            k, v, h, do, dh,
            T, D,
            BT=BT, BD=BD,
            num_warps=num_warps
        )

        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        grid_dqkv = (triton.cdiv(D, BD), triton.cdiv(T, BT), B * H)
        chunk_retention_bwd_kernel_dqkv[grid_dqkv](
            q, k, v, h,
            do, dq, dk, dv,
            T, D,
            BT=BT, BD=BD,
            num_warps=num_warps
        )

        if initial_state is not None:
            dk[:, :, 0] += (initial_state * dh[:, :, 0] * tl.exp(-tl.exp(k[:, :, 0])) * tl.exp(k[:, :, 0])).sum(dim=-1)

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None

def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkRetentionFunction.apply(q, k, v, initial_state, output_final_state)
    return o, final_state
