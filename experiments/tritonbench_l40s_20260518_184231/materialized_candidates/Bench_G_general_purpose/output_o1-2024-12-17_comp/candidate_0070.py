import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------------------
# Forward kernel for hidden states
# -----------------------------------------------------------------------------------------
@triton.jit
def chunk_retention_fwd_kernel_h(
    Q_ptr, K_ptr, V_ptr, H_ptr,  # pointers to Q, K, V, and H
    H_in_ptr, H_out_ptr,        # pointers to optional initial and final hidden states
    B, T, C, BT,                # batch, time, channel, chunk size
    scale,                      # floating scale factor
    boundary_check,             # boundary check (0 or 1)
    BLOCK_SIZE: tl.constexpr    # block size for the kernel
):
    pid = tl.program_id(0)
    # Each program_id handles one chunk of size BT
    chunk_start = pid * BT
    # Load batch index from program_id(1) if used in a 2D launch grid
    # For now, assume single-dim grid, or that b_idx is handled externally
    b_idx = 0

    offs_c = tl.arange(0, BLOCK_SIZE)
    # chunk boundaries
    chunk_end = chunk_start + BT

    # check boundary
    if boundary_check == 1:
        is_valid = chunk_start < T
    else:
        is_valid = 1

    # If chunk is valid, proceed
    if is_valid != 0:
        # optionally load initial hidden state
        # In actual code, boundary checks for offset sizes must be applied
        if chunk_start == 0 and H_in_ptr != 0:
            # load from H_in_ptr
            h_init = tl.load(H_in_ptr + b_idx * C + offs_c, mask=offs_c < C)
        else:
            # set initial to zero
            h_init = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # main loop over each step in the chunk
        h_curr = h_init
        for t in range(chunk_start, chunk_end):
            # boundary check
            if t >= T:
                break

            # load q, k, v
            q = tl.load(Q_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            k = tl.load(K_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            v = tl.load(V_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)

            # compute hidden state update: h_curr = h_curr + scale * (q * k) * v
            # (example logic, can be replaced by actual chunk retention formula)
            kv = tl.broadcast_to(q * k, [BLOCK_SIZE]) * scale
            h_curr = h_curr + kv * v

            # store h_curr in H
            out_idx = b_idx * T * C + t * C + offs_c
            tl.store(H_ptr + out_idx, h_curr, mask=offs_c < C)

        # optionally store final hidden state
        if chunk_end >= T and H_out_ptr != 0:
            tl.store(H_out_ptr + b_idx * C + offs_c, h_curr, mask=offs_c < C)


def chunk_fwd_h_fn(q, k, v, h, h_in, h_out, bt, scale, boundary_check):
    B, T, C = q.shape
    # grid
    # each chunk is size bt, so number of chunks = ceil(T / bt)
    n_chunks = (T + bt - 1) // bt

    block_size = triton.next_power_of_2(C)
    grid = (n_chunks,)

    # If optional h_in or h_out is None, pass 0
    h_in_ptr = h_in.data_ptr() if (h_in is not None) else 0
    h_out_ptr = h_out.data_ptr() if (h_out is not None) else 0

    chunk_retention_fwd_kernel_h[grid](
        q, k, v, h,
        h_in_ptr, h_out_ptr,
        B, T, C, bt,
        scale,
        boundary_check,
        BLOCK_SIZE=block_size
    )


# -----------------------------------------------------------------------------------------
# Forward kernel for final output
# -----------------------------------------------------------------------------------------
@triton.jit
def chunk_retention_fwd_kernel_o(
    Q_ptr, K_ptr, V_ptr, H_ptr, O_ptr,
    B, T, C, BT,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    chunk_start = pid * BT
    b_idx = 0

    offs_c = tl.arange(0, BLOCK_SIZE)
    chunk_end = chunk_start + BT

    if chunk_start < T:
        for t in range(chunk_start, chunk_end):
            if t >= T:
                break
            q = tl.load(Q_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            k = tl.load(K_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            v = tl.load(V_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            h = tl.load(H_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)

            # compute final output: o = h + scale*(q*k)
            # (example logic, can be replaced by actual chunk retention formula)
            val = h + (q * k) * scale
            tl.store(O_ptr + b_idx * T * C + t * C + offs_c, val, mask=offs_c < C)


def chunk_fwd_o_fn(q, k, v, h, o, bt, scale):
    B, T, C = q.shape
    n_chunks = (T + bt - 1) // bt
    block_size = triton.next_power_of_2(C)
    grid = (n_chunks,)

    chunk_retention_fwd_kernel_o[grid](
        q, k, v, h, o,
        B, T, C, bt,
        scale,
        BLOCK_SIZE=block_size
    )


# -----------------------------------------------------------------------------------------
# Backward kernel for hidden states
# -----------------------------------------------------------------------------------------
@triton.jit
def chunk_retention_bwd_kernel_dh(
    Q_ptr, DO_ptr, DH_ptr,
    B, T, C, BT,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    chunk_start = pid * BT
    b_idx = 0
    offs_c = tl.arange(0, BLOCK_SIZE)
    chunk_end = chunk_start + BT

    if chunk_start < T:
        # example: dh = do + scale * q
        # loop over chunk
        for t in range(chunk_start, chunk_end):
            if t >= T:
                break
            do_val = tl.load(DO_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            q_val = tl.load(Q_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            dh_val = do_val + q_val * scale
            tl.store(DH_ptr + b_idx * T * C + t * C + offs_c, dh_val, mask=offs_c < C)


def chunk_bwd_dh_fn(q, do, dh, bt, scale):
    B, T, C = q.shape
    n_chunks = (T + bt - 1) // bt
    block_size = triton.next_power_of_2(C)
    grid = (n_chunks,)

    chunk_retention_bwd_kernel_dh[grid](
        q, do, dh,
        B, T, C, bt,
        scale,
        BLOCK_SIZE=block_size
    )


# -----------------------------------------------------------------------------------------
# Backward kernel for gradients wrt Q, K, V
# -----------------------------------------------------------------------------------------
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    DO_ptr, DH_ptr, H_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    B, T, C, BT,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    chunk_start = pid * BT
    b_idx = 0
    offs_c = tl.arange(0, BLOCK_SIZE)
    chunk_end = chunk_start + BT

    if chunk_start < T:
        # example: dq = do + dh, dk = do * h, dv = dh * scale
        for t in range(chunk_start, chunk_end):
            if t >= T:
                break
            do_val = tl.load(DO_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            dh_val = tl.load(DH_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)
            h_val = tl.load(H_ptr + b_idx * T * C + t * C + offs_c, mask=offs_c < C)

            dq_val = do_val + dh_val
            dk_val = do_val * h_val
            dv_val = dh_val * scale

            tl.store(DQ_ptr + b_idx * T * C + t * C + offs_c, dq_val, mask=offs_c < C)
            tl.store(DK_ptr + b_idx * T * C + t * C + offs_c, dk_val, mask=offs_c < C)
            tl.store(DV_ptr + b_idx * T * C + t * C + offs_c, dv_val, mask=offs_c < C)


def chunk_bwd_dqkv_fn(do, dh, h, dq, dk, dv, bt, scale):
    B, T, C = do.shape
    n_chunks = (T + bt - 1) // bt
    block_size = triton.next_power_of_2(C)
    grid = (n_chunks,)

    chunk_retention_bwd_kernel_dqkv[grid](
        do, dh, h,
        dq, dk, dv,
        B, T, C, bt,
        scale,
        BLOCK_SIZE=block_size
    )


# -----------------------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------------------
class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, h_in, scale, bt, boundary_check):
        B, T, C = q.shape
        h = torch.empty_like(q)
        o = torch.empty_like(q)

        h_out = None
        if h_in is not None:
            h_out = torch.empty_like(h_in)

        # Forward hidden states
        chunk_fwd_h_fn(q, k, v, h, h_in, h_out, bt, scale, boundary_check)
        # Forward output
        chunk_fwd_o_fn(q, k, v, h, o, bt, scale)

        # Save for backward
        ctx.save_for_backward(q, k, v, h, o, h_in, h_out)
        ctx.scale = scale
        ctx.bt = bt
        ctx.boundary_check = boundary_check
        return o, h, h_out

    @staticmethod
    def backward(ctx, do, dh, dh_out):
        q, k, v, h, o, h_in, h_out = ctx.saved_tensors
        scale = ctx.scale
        bt = ctx.bt
        boundary_check = ctx.boundary_check

        # Gradient wrt Q, K, V
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        # Gradient wrt hidden states
        dh_new = torch.zeros_like(h)

        # 1) Compute gradient wrt hidden states
        chunk_bwd_dh_fn(q, do, dh_new, bt, scale)
        # accumulate with dh from upstream
        dh_new.add_(dh)

        # 2) Compute gradient wrt Q, K, V
        chunk_bwd_dqkv_fn(do, dh_new, h, dq, dk, dv, bt, scale)

        # No gradient wrt scale, h_in, boundary_check out of simplicity
        dscale = None
        dh_in = None
        return dq, dk, dv, dh_in, dscale, None, None


# -----------------------------------------------------------------------------------------
# Public function
# -----------------------------------------------------------------------------------------
def chunk_retention(q, k, v, h_in=None, scale=1.0, bt=128, boundary_check=0):
    """
    q, k, v: (B, T, C)
    h_in: optional hidden state (B, C) for initialization
    scale: float
    bt: chunk size
    boundary_check: integer
    """
    return ChunkRetentionFunction.apply(q, k, v, h_in, scale, bt, boundary_check)
