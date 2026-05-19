import torch
import triton
import triton.language as tl
import math

@triton.jit
def attention_fwd_kernel(
    Q, K, V, H, O,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_hb, stride_hh, stride_ht, stride_hd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, T, D,
    scale,
    IFCOND: tl.constexpr, STORE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Identify the batch and head using the program ids
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Compute pointers to the base of each input in memory
    q_ptr = Q + batch_id * stride_qb + head_id * stride_qh
    k_ptr = K + batch_id * stride_kb + head_id * stride_kh
    v_ptr = V + batch_id * stride_vb + head_id * stride_vh
    h_ptr = H + batch_id * stride_hb + head_id * stride_hh
    o_ptr = O + batch_id * stride_ob + head_id * stride_oh

    # Offsets for each thread program in T and D dimensions
    off_t = tl.arange(0, BLOCK_T)
    off_d = tl.arange(0, BLOCK_D)
    # b_h is to store intermediate results, initialize to zeros
    b_h = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    # Loop over the sequence length in blocks
    for t_block in range(0, T, BLOCK_T):
        # Compute pointers for the block in Q, K, V
        q_block_ptr = q_ptr + t_block * stride_qt
        k_block_ptr = k_ptr + t_block * stride_kt
        v_block_ptr = v_ptr + t_block * stride_vt

        # Load q and k block
        # Each is (BLOCK_T x BLOCK_D)
        q_block = tl.load(
            q_block_ptr + off_t[:, None] * stride_qt + off_d[None, :] * stride_qd,
            mask=(off_t[:, None] + t_block < T) & (off_d[None, :] < D),
            other=0.0
        )
        k_block = tl.load(
            k_block_ptr + off_t[:, None] * stride_kt + off_d[None, :] * stride_kd,
            mask=(off_t[:, None] + t_block < T) & (off_d[None, :] < D),
            other=0.0
        )

        # Scaled dot product for the block
        b_s = tl.sum(q_block * k_block, axis=1) * scale

        # Expand b_s for elementwise multiplication with V
        b_s = b_s[:, None]

        # Load v block
        v_block = tl.load(
            v_block_ptr + off_t[:, None] * stride_vt + off_d[None, :] * stride_vd,
            mask=(off_t[:, None] + t_block < T) & (off_d[None, :] < D),
            other=0.0
        )

        # Weighted V => partial output block
        b_o = b_s * v_block

        # Depending on IFCOND, perform a conditional or standard update
        if IFCOND:
            b_h = tl.where(b_o > b_h, b_o, b_h)
        else:
            b_h += b_o

        # If STORE is set, optionally store intermediate b_h in H
        if STORE:
            tl.store(
                h_ptr + (off_t[:, None] + t_block) * stride_ht + off_d[None, :] * stride_hd,
                b_h,
                mask=(off_t[:, None] + t_block < T) & (off_d[None, :] < D)
            )

    # Finally, store the accumulated b_h in the output O
    tl.store(
        o_ptr + off_t[:, None] * stride_ot + off_d[None, :] * stride_od,
        b_h,
        mask=(off_t[:, None] < T) & (off_d[None, :] < D)
    )

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, IFCOND=False, STORE=False):
        B, H, T, D = q.shape
        o = torch.empty_like(q)
        h = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D)

        # Grid for batch and head
        grid = (B, H)
        BLOCK_T = 128
        BLOCK_D = 64

        attention_fwd_kernel[grid](
            q, k, v, h, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            h.stride(0), h.stride(1), h.stride(2), h.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, T, D,
            scale,
            IFCOND, STORE,
            BLOCK_T, BLOCK_D,
            num_warps=4,
            num_stages=2
        )
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # This example does not implement backward pass.
        raise NotImplementedError("Backward pass is not implemented.")
