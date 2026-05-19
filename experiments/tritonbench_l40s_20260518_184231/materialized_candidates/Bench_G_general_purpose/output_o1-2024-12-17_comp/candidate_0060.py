import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    B, M, N,
    stride_qb, stride_qm, stride_qn,
    stride_kb, stride_km, stride_kn,
    stride_cs,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Program IDs for batch (pid_b) and row block within head dimension (pid_m)
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Offsets within the Q/K memory
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers for Q, K, cos, sin for current batch
    q_offs = pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    k_offs = pid_b * stride_kb + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn

    # Load Q, K from memory
    q = tl.load(q_ptr + q_offs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    k = tl.load(k_ptr + k_offs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # We'll split the dimension (N) into two halves
    half_n = N // 2

    # Load cos, sin
    # For a given row offset (offs_m), we assume each corresponds to an index into cos/sin.
    # Adjust offset for half-rotary embedding usage if needed.
    cos = tl.load(cos_ptr + (offs_m % half_n) * stride_cs, mask=(offs_m < M), other=1.0)
    sin = tl.load(sin_ptr + (offs_m % half_n) * stride_cs, mask=(offs_m < M), other=0.0)

    # Broadcast to match the tile shape
    cos = cos[:, None]
    sin = sin[:, None]

    # Identify even/odd indices
    even_mask = offs_n % 2 == 0
    odd_mask = ~even_mask

    # Extract the parts of Q, K for rotation
    q_even = tl.where(even_mask, q, 0.0)
    q_odd = tl.where(odd_mask, q, 0.0)
    k_even = tl.where(even_mask, k, 0.0)
    k_odd = tl.where(odd_mask, k, 0.0)

    # Apply rotation
    # Forward pass: new_even = even * cos - odd * sin, new_odd = odd * cos + even * sin
    # Backward pass: new_even = even * cos + odd * sin, new_odd = odd * cos - even * sin
    # We switch the sign of sin to invert rotation on backward pass
    if BACKWARD_PASS:
        new_q_even = q_even * cos + q_odd * sin
        new_q_odd = q_odd * cos - q_even * sin
        new_k_even = k_even * cos + k_odd * sin
        new_k_odd = k_odd * cos - k_even * sin
    else:
        new_q_even = q_even * cos - q_odd * sin
        new_q_odd = q_odd * cos + q_even * sin
        new_k_even = k_even * cos - k_odd * sin
        new_k_odd = k_odd * cos + k_even * sin

    # Combine even/odd positions
    q_rot = tl.where(even_mask, new_q_even, new_q_odd)
    k_rot = tl.where(even_mask, new_k_even, new_k_odd)

    # Store results
    tl.store(q_ptr + q_offs, q_rot, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    tl.store(k_ptr + k_offs, k_rot, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def rope_backward(
    dq, dk,
    cos, sin,
    B, M, N,
    stride_qb, stride_qm, stride_qn,
    stride_kb, stride_km, stride_kn,
    stride_cs
):
    grid = (B, (M + 31) // 32)  # Example grid: parallelize over batch and row blocks
    triton.run(
        _triton_rope,
        q_ptr=dq, k_ptr=dk,
        cos_ptr=cos, sin_ptr=sin,
        B=B, M=M, N=N,
        stride_qb=stride_qb, stride_qm=stride_qm, stride_qn=stride_qn,
        stride_kb=stride_kb, stride_km=stride_km, stride_kn=stride_kn,
        stride_cs=stride_cs,
        BACKWARD_PASS=True,
        BLOCK_M=32,  # tuning parameters
        BLOCK_N=32,  # tuning parameters
        grid=grid
    )
