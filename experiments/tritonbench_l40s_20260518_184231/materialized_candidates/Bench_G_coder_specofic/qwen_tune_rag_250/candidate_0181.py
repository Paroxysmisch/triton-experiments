(0, BLOCK_SIZE_K // 2) * 2

    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * (offs_n[None, :] + 1))
    x_imag = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0)

    offs_cn = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * K + pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2) * 2
    cos, sin = get_freq_multi_tokens(offs_cn, start_token_position, THETA, BLOCK_SIZE_M)
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    out_ptrs = out_ptr + (pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    tl.store(out_ptrs, out_real, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K))
    out_ptrs = out_ptr + (pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * (offs_n[None, :] + 1))
    tl.store(out_ptrs, out_imag, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K))


def rbe_triton_wrapper(x, start_token_position=0, theta=10000.):
    batch, M, K = x.shape
    out = torch.empty_like(x)
    grid = lambda META: (batch, triton.cdiv(M * K, META['BLOCK_SIZE_M'] * META['BLOCK_SIZE_K']) * 2)
    rbe_triton[grid](x, out,
                     M, K,
                     *x.stride(),
                     *out.stride(),
                     start_token_position=start_token_position,
                     THETA=theta, BLOCK_SIZE_M=16, BLOCK_SIZE_K=16,
                     )
    return out


@triton.jit
def rms_matmul_rbe(x_ptr, w_ptr, rms_w_ptr, out_ptr,
                   M, N, K,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_w_batch, stride_w_n, stride_w_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_n,
                   start_token_position, USE_FP8: tl.constexpr, RBE_EPILOGUE: tl.constexpr, THETA: tl.constexpr, EPS: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = pid_batch * stride_x_batch + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_batch * stride_w_batch + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_m + stride_x_k * tl.arange(0, BLOCK_SIZE_K))
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (tl.arange(0, BLOCK_SIZE_K) < K), other=0.0)

    w_ptrs = w_ptr + (offs_n[None, :] * stride_w_n + stride_w_k * tl.arange(0, BLOCK_SIZE_K))
    w = tl.load(w_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K) < K) & (offs_n[None, :] < N), other=0.0)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K) * stride_rms_w
    rms_w = tl.load(rms_w_ptrs)

    if USE_FP8:
        x = x.to(tl.float8e5)
        w = w.to(tl.float8e5)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_k = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_m + x_k[:, None] * stride_x_k)
        x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (x_k[:, None] < K), other=0.0).to(tl.float32)
        w_ptrs = w_ptr + (x_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
        w_block = tl.load(w_ptrs, mask=(x_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)

        rms_w_ptrs = rms_w_ptr + x_k * stride_rms_w
        rms_w_block = tl.load(rms_w_ptrs, mask=x_k < K, other=0.0)

        x_hat = x_block * rms_w_block[None, :]
        acc += tl.dot(x_hat, w_block)

    if RBE_EPILOGUE:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        cos, sin = get_freq_multi_tokens(offs_cn, start_token_position, THETA, N)
        acc_real = acc[:, :, None] * cos[None, None, :] - acc[:, :, None] * sin[None, None, :]
        acc_real = acc_real.sum(1)
        tl.store(out_ptr + pid_m * stride_out_m + pid_n * stride_out_n + tl.arange(0, BLOCK_SIZE_M) * stride_out_batch,
                 acc_real)
    else:
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_ptrs = out_ptr + stride_out_batch * pid_batch + offs_m[:, None] * stride_out_m + stride_out_n * offs_n[None, :]
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def rms_matmul_rbe_qkv(x, q_weight, q_rms_weight, k_weight, k_rms_weight, v_weight, v_rms_weight, start_token_position=0, theta=10000., fp8=False):
    def rms_matmul_rbe_qkv_epilogue(x, q_weight, q_rms_weight, k_weight, k_rms_weight, v_weight, v_rms_weight, start_token_position, theta, fp8):
        q = rms_matmul_rbe_qkv_wrapper(x, q_weight, q_rms_weight, start_token_position=start_token_position, theta=theta, fp8=fp8)
        q = rbe_triton_wrapper(q, start_token_position=start_token_position, theta=theta)
        k = rms_matmul_rbe_qkv_wrapper(x, k_weight, k_rms_weight, start_token_position=start_token_position, theta=theta, fp8=fp8)
        v = rms_matmul_rbe_qkv_wrapper(x, v_weight, v_rms_weight, start_token_position=start_token_position, theta=theta, fp8=fp8)
        return q, k, v

    return rms_matmul_rbe_qkv_epilogue


@torch.inference_mode()
def rms_matmul_rbe_qkv_wrapper(x, weight, rms_weight, eps=1e-6, start_token_position=0, theta=10000., fp8=False):
    batch, M, K = x.shape
    assert weight.shape[-1] == K
    out = torch.empty((batch, M, K), dtype=x.dtype, device=x.device)
    q_k_v_grid = lambda META: (batch, triton.cdiv(M, META['BLOCK_SIZE_M']), 3)
    rms_matmul_rbe[q_k_v_grid](x, weight, rms_weight, out,
                               M, K, K,
                               *x.stride(),
                               *weight.stride(),
                               *rms_weight.stride(),
                               *out.stride(),
                               start_token_position=start_token_position,
                               USE_FP8=fp8, RBE_EPILOGUE=False, THETA=theta, EPS=eps,
                               BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
                               )
    return out
