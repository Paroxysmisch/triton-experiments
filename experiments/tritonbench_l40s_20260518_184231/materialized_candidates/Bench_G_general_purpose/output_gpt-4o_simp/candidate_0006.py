import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(Q, K, V, B0, Out, 
                        stride_qz, stride_qh, stride_qm, stride_qk, 
                        stride_kz, stride_kh, stride_kn, stride_kk, 
                        stride_vz, stride_vh, stride_vn, stride_vk, 
                        stride_b0z, stride_b0h, stride_b0m, stride_b0n, 
                        stride_outz, stride_outh, stride_outm, stride_outk,
                        start_m, off_hz, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Load input blocks
    pid_m = tl.program_id(0) + start_m
    pid_hz = tl.program_id(1) + off_hz

    # Compute the position in the batch and the sequence
    q_offset = pid_hz * stride_qh + pid_m * BLOCK_M * stride_qm
    k_offset = pid_hz * stride_kh
    v_offset = pid_hz * stride_vh
    b0_offset = pid_hz * stride_b0h + pid_m * BLOCK_M * stride_b0m

    # Load Q, K, V, B0 blocks
    Q_block = tl.load(Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_N)[None, :] * stride_qk)
    K_block = tl.load(K + k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_kn + tl.arange(0, BLOCK_N)[None, :] * stride_kk)
    V_block = tl.load(V + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, BLOCK_N)[None, :] * stride_vk)
    B0_block = tl.load(B0 + b0_offset + tl.arange(0, BLOCK_M)[:, None] * stride_b0m + tl.arange(0, BLOCK_N)[None, :] * stride_b0n)

    # Compute scaled dot-product attention
    scale = 1.0 / tl.sqrt(tl.float32(BLOCK_N))
    logits = tl.dot(Q_block, K_block) * scale + B0_block
    weights = tl.softmax(logits, axis=1)

    # Compute output
    Out_block = tl.dot(weights, V_block)

    # Store the result
    out_offset = pid_hz * stride_outh + pid_m * BLOCK_M * stride_outm
    tl.store(Out + out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_outm + tl.arange(0, BLOCK_N)[None, :] * stride_outk, Out_block)

def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, Out, BLOCK_M=128, BLOCK_N=64):
    # Define grid dimensions
    grid = (Q.shape[1] // BLOCK_M, Q.shape[0])  # Assuming Q.shape = (batch_size, seq_len, dim)

    # Launch kernel
    triton.kernel(
        _fwd_kernel_aligned,
        grid=grid,
        num_warps=4,  # Choose appropriate number of warps
        args=[Q, K, V, B0, Out, 
              Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
              K.stride(0), K.stride(1), K.stride(2), K.stride(3),
              V.stride(0), V.stride(1), V.stride(2), V.stride(3),
              B0.stride(0), B0.stride(1), B0.stride(2), B0.stride(3),
              Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
              0, 0, BLOCK_M, BLOCK_N],
        stream=triton.Stream()
    )
