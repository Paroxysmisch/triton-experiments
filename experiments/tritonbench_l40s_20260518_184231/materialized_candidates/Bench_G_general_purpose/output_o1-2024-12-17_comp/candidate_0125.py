import torch
import triton
import triton.language as tl


# --------------------------------------------
# Forward Kernel
# --------------------------------------------
@triton.jit
def parallel_rebased_fwd_kernel(
    Q_PTR, K_PTR, V_PTR,
    Z_PTR, O_PTR,
    BATCH, SEQ_LEN, D_MODEL,
    STRIDE_QB, STRIDE_QS, STRIDE_QD,
    STRIDE_KB, STRIDE_KS, STRIDE_KD,
    STRIDE_VB, STRIDE_VS, STRIDE_VD,
    STRIDE_ZB, STRIDE_ZS,
    STRIDE_OB, STRIDE_OS, STRIDE_OD,
    USE_SCALE, USE_NORMALIZE,
    BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Offsets
    # Block offsets in query, key, value
    q_start = batch_idx * STRIDE_QB + seq_idx * BLOCK_SIZE * STRIDE_QS
    # We'll loop over K blocks
    k_blocks = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Registers for partial sums, z-values, etc.
    out_o = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    out_z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over blocks of K
    for kb in range(k_blocks):
        # Offsets for K and V
        k_start = batch_idx * STRIDE_KB + kb * BLOCK_SIZE * STRIDE_KS
        v_start = batch_idx * STRIDE_VB + kb * BLOCK_SIZE * STRIDE_VS

        # Load Q
        q_idx = tl.arange(0, BLOCK_SIZE)
        d_idx = tl.arange(0, BLOCK_SIZE)
        q_ptrs = Q_PTR + q_start + q_idx[:, None] * STRIDE_QS + d_idx[None, :] * STRIDE_QD
        q_block = tl.load(q_ptrs, mask=(q_idx[:, None] < SEQ_LEN) & (d_idx[None, :] < D_MODEL), other=0.0)

        # Load K
        k_ptrs = K_PTR + k_start + d_idx[:, None] * STRIDE_KS + tl.arange(0, BLOCK_SIZE)[None, :] * STRIDE_KD
        k_block = tl.load(k_ptrs, mask=(d_idx[:, None] < D_MODEL) & (kb*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] < SEQ_LEN), other=0.0)

        # QK^T
        score = tl.dot(q_block, k_block)

        # Scale if needed
        if USE_SCALE:
            scale_val = 1.0 / tl.sqrt(tl.float32(D_MODEL))
            score = score * scale_val

        # Accumulate partial scores
        if USE_NORMALIZE:
            max_score = tl.max(score, 1)
            score = score - max_score[:, None]
            numer = tl.exp(score)
            denom = tl.sum(numer, 1)
            out_z = out_z + denom
            # Multiply partial result by V in next loop
            # we'll do this after we've fully computed all partial scores,
            # but for demonstration we'll just do partial operation here
            # (for actual code we might restructure)
            # Load V
            v_ptrs = V_PTR + v_start + tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE_VS + tl.arange(0, BLOCK_SIZE)[None, :] * STRIDE_VD
            v_block = tl.load(v_ptrs, mask=(kb*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < SEQ_LEN) & (tl.arange(0, BLOCK_SIZE)[None, :] < D_MODEL), other=0.0)
            out_o = out_o + tl.dot(numer, v_block)  # partial
        else:
            # No normalization
            # Just do partial O accumulation
            v_ptrs = V_PTR + v_start + tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE_VS + tl.arange(0, BLOCK_SIZE)[None, :] * STRIDE_VD
            v_block = tl.load(v_ptrs, mask=(kb*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < SEQ_LEN) & (tl.arange(0, BLOCK_SIZE)[None, :] < D_MODEL), other=0.0)
            out_o = out_o + tl.dot(score, v_block)

    # Write Z (if normalized)
    if USE_NORMALIZE:
        z_ptrs = Z_PTR + batch_idx * STRIDE_ZB + seq_idx * BLOCK_SIZE * STRIDE_ZS + tl.arange(0, BLOCK_SIZE) * STRIDE_ZS
        tl.store(z_ptrs, out_z, mask=(seq_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < SEQ_LEN))

    # Write O
    o_ptrs = O_PTR + batch_idx * STRIDE_OB + seq_idx * BLOCK_SIZE * STRIDE_OS
    for i in range(BLOCK_SIZE):
        row_mask = seq_idx * BLOCK_SIZE + i < SEQ_LEN
        out_row = out_o[i, :]
        out_ptrs = o_ptrs + i * STRIDE_OS + tl.arange(0, BLOCK_SIZE) * STRIDE_OD
        tl.store(out_ptrs, out_row.to(Q_PTR.dtype.element_ty), mask=(tl.arange(0, BLOCK_SIZE) < D_MODEL) & row_mask)


# --------------------------------------------
# Backward Helper: dQ
# --------------------------------------------
@triton.jit
def _parallel_rebased_bwd_dq(
    DO_PTR, DZ_PTR, K_PTR,
    DQ_PTR,
    BATCH, SEQ_LEN, D_MODEL,
    STRIDE_DO_B, STRIDE_DO_S, STRIDE_DO_D,
    STRIDE_DZ_B, STRIDE_DZ_S,
    STRIDE_KB, STRIDE_KS, STRIDE_KD,
    STRIDE_DQ_B, STRIDE_DQ_S, STRIDE_DQ_D,
    USE_SCALE, USE_NORMALIZE,
    BLOCK_SIZE: tl.constexpr
):
    # Implementation for dq partial grad
    # Program IDs
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Offsets
    do_start = batch_idx * STRIDE_DO_B + seq_idx * BLOCK_SIZE * STRIDE_DO_S
    dz_start = batch_idx * STRIDE_DZ_B + seq_idx * BLOCK_SIZE * STRIDE_DZ_S

    # We'll do a simple partial approach
    dq_out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for kb in range((SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE):
        do_ptrs = DO_PTR + do_start
        # load partial do
        do_idx = tl.arange(0, BLOCK_SIZE)
        d_idx = tl.arange(0, BLOCK_SIZE)
        do_val = tl.load(
            do_ptrs + do_idx[:, None] * STRIDE_DO_S + d_idx[None, :] * STRIDE_DO_D,
            mask=(do_idx[:, None] < SEQ_LEN) & (d_idx[None, :] < D_MODEL),
            other=0.0,
        )
        # Possibly combine with DZ if normalization
        if USE_NORMALIZE:
            dz_val = tl.load(
                DZ_PTR + dz_start + do_idx * STRIDE_DZ_S,
                mask=(seq_idx*BLOCK_SIZE + do_idx < SEQ_LEN),
                other=0.0,
            )
            # simplified example usage of dz
            do_val = do_val * dz_val[:, None]

        # load K
        k_start = batch_idx * STRIDE_KB + kb * BLOCK_SIZE * STRIDE_KS
        k_ptrs = K_PTR + k_start + d_idx[:, None] * STRIDE_KS + tl.arange(0, BLOCK_SIZE)[None, :] * STRIDE_KD
        k_val = tl.load(
            k_ptrs,
            mask=(d_idx[:, None] < D_MODEL) & (kb*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] < SEQ_LEN),
            other=0.0,
        )
        dq_out += tl.dot(do_val, k_val)

    # scale if needed
    if USE_SCALE:
        dq_out *= (1.0 / tl.sqrt(tl.float32(D_MODEL)))

    # Store
    dq_ptrs = DQ_PTR + batch_idx * STRIDE_DQ_B + seq_idx * BLOCK_SIZE * STRIDE_DQ_S
    for i in range(BLOCK_SIZE):
        row_mask = seq_idx * BLOCK_SIZE + i < SEQ_LEN
        out_row = dq_out[i, :]
        out_ptrs = dq_ptrs + i * STRIDE_DQ_S + tl.arange(0, BLOCK_SIZE) * STRIDE_DQ_D
        tl.store(out_ptrs, out_row.to(DQ_PTR.dtype.element_ty), mask=(tl.arange(0, BLOCK_SIZE) < D_MODEL) & row_mask)


# --------------------------------------------
# Backward Helper: dK, dV
# --------------------------------------------
@triton.jit
def _parallel_rebased_bwd_dkv(
    DO_PTR, DZ_PTR, Q_PTR,
    DK_PTR, DV_PTR,
    BATCH, SEQ_LEN, D_MODEL,
    STRIDE_DO_B, STRIDE_DO_S, STRIDE_DO_D,
    STRIDE_DZ_B, STRIDE_DZ_S,
    STRIDE_QB, STRIDE_QS, STRIDE_QD,
    STRIDE_DK_B, STRIDE_DK_S, STRIDE_DK_D,
    STRIDE_DV_B, STRIDE_DV_S, STRIDE_DV_D,
    USE_SCALE, USE_NORMALIZE,
    BLOCK_SIZE: tl.constexpr
):
    # Program IDs
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Offsets
    do_start = batch_idx * STRIDE_DO_B + seq_idx * BLOCK_SIZE * STRIDE_DO_S
    dz_start = batch_idx * STRIDE_DZ_B + seq_idx * BLOCK_SIZE * STRIDE_DZ_S
    q_start = batch_idx * STRIDE_QB

    dk_out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    dv_out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Load partial DO
    do_idx = tl.arange(0, BLOCK_SIZE)
    d_idx = tl.arange(0, BLOCK_SIZE)
    do_val = tl.load(
        DO_PTR + do_start + do_idx[:, None] * STRIDE_DO_S + d_idx[None, :] * STRIDE_DO_D,
        mask=(seq_idx*BLOCK_SIZE + do_idx[:, None] < SEQ_LEN) & (d_idx[None, :] < D_MODEL),
        other=0.0,
    )

    # Possibly scale with Z
    if USE_NORMALIZE:
        dz_val = tl.load(
            DZ_PTR + dz_start + do_idx * STRIDE_DZ_S,
            mask=(seq_idx*BLOCK_SIZE + do_idx < SEQ_LEN),
            other=0.0,
        )
        do_val = do_val * dz_val[:, None]

    # Loop over Q blocks
    for qb in range((SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE):
        # load Q
        q_block_start = q_start + qb * BLOCK_SIZE * STRIDE_QS
        q_ptrs = Q_PTR + q_block_start + do_idx[:, None] * STRIDE_QS + d_idx[None, :] * STRIDE_QD
        q_val = tl.load(
            q_ptrs,
            mask=(qb*BLOCK_SIZE + do_idx[:, None] < SEQ_LEN) & (d_idx[None, :] < D_MODEL),
            other=0.0,
        )
        # accumulate dK
        dk_out += tl.dot(q_val.transpose([1,0]), do_val).transpose([1,0])
        # accumulate dV
        dv_out += tl.dot(do_val.transpose([1,0]), q_val).transpose([1,0])

    # scale if needed
    if USE_SCALE:
        dk_out *= (1.0 / tl.sqrt(tl.float32(D_MODEL)))

    # Store dK
    dk_ptrs = DK_PTR + batch_idx * STRIDE_DK_B + seq_idx * BLOCK_SIZE * STRIDE_DK_S
    for i in range(BLOCK_SIZE):
        row_mask = seq_idx * BLOCK_SIZE + i < SEQ_LEN
        out_row = dk_out[i, :]
        out_ptrs = dk_ptrs + i * STRIDE_DK_S + tl.arange(0, BLOCK_SIZE) * STRIDE_DK_D
        tl.store(out_ptrs, out_row.to(DK_PTR.dtype.element_ty), mask=(tl.arange(0, BLOCK_SIZE) < D_MODEL) & row_mask)

    # Store dV
    dv_ptrs = DV_PTR + batch_idx * STRIDE_DV_B + seq_idx * BLOCK_SIZE * STRIDE_DV_S
    for i in range(BLOCK_SIZE):
        row_mask = seq_idx *
