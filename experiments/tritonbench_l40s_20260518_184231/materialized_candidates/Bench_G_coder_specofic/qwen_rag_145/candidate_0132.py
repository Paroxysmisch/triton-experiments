@triton.jit
def _fwd_kernel(
    Q, K, V, O, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    M, N, DMODEL,
    sm_scale, 
    IS_CAUSAL: tl.constexpr, 
    USE_FP8: tl.constexpr
):
    # Define local variables.
    pid_m, pid_n, pid_dm = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset_m, offset_n = pid_m * BLOCK_M, pid_n * BLOCK_N
    offset_dm = pid_dm * BLOCK_DMODEL
    mask = pid_n < pid_dm < N
    # Load matrices.
    Q_part = tl.load(Q + offset_m + offset_n + offset_dm, mask=mask)
    K_part = tl.load(K + offset_m + offset_n + offset_dm, mask=mask)
    V_part = tl.load(V + offset_m + offset_n + offset_dm, mask=mask)
    # Compute scaled dot-product attention.
    dot_product = tl.dot(Q_part, K_part, trans_b=True) * sm_scale
    if USE_FP8:
        dot_product = dot_product.to(tl.int8)
    softmax_temp = tl.softmax(dot_product, axis=-1)
    # If causal, mask future positions.
    if IS_CAUSAL:
        mask_future = tl.tril(tl.ones_like(softmax_temp), diagonal=1)
        softmax_temp = softmax_temp * mask_future
    # Compute output.
    attn_output = tl.dot(softmax_temp, V_part)
    # Store result.
    tl.store(O + offset_m + offset_n + offset_dm, attn_output, mask=mask)

def triton_fa(
    Q, K, V, sm_scale, 
    IS_CAUSAL=False, 
    USE_FP8=False, 
    BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=128
):
    # Check data type consistency.
    assert Q.dtype == K.dtype == V.dtype == O.dtype
    # Compute grid and block dimensions.
    M, N, DMODEL = Q.shape
    grid = (triton.cdiv(M, BLOCK_M), N, DMODEL)
    # Invoke kernel.
    _fwd_kernel[grid](
        Q, K, V, O, 
        BLOCK_M, BLOCK_N, BLOCK_DMODEL,
        M, N, DMODEL,
        sm_scale, 
        IS_CAUSAL=IS_CAUSAL, 
        USE_FP8=USE_FP8
    )
