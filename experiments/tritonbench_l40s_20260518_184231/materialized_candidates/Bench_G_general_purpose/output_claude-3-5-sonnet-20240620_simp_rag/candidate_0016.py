import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    N, K,
    lora_indices,
    scaling,
    xm_stride, xk_stride, wm_stride, wk_stride, om_stride,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_N)
    num_pid_k = SPLIT_K
    num_pid_in_group = num_pid_k * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_k * num_pid_m
    pid_m = (pid % group_size) % num_pid_m
    pid_k = (pid % group_size) // num_pid_m

    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < N
    mask_k = offs_k < K

    lora_index = tl.load(lora_indices + group_id)
    x_ptrs = input_ptr + group_id * xm_stride + offs_m[:, None] * xk_stride + offs_k[None, :] * 1
    w_ptrs = lora_ptr + lora_index * wm_stride + offs_m[:, None] * wk_stride + offs_k[None, :] * 1

    acc = tl.zeros((BLOCK_N, 1), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_k[None, :] & mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[None, :] & mask_m[:, None], other=0.0)
        acc += tl.sum(x * w, axis=1)[:, None]

    acc = acc * scaling
    o_ptrs = out_ptr + group_id * om_stride + offs_m * 1
    if SPLIT_K == 1:
        tl.store(o_ptrs, acc.squeeze(1), mask=mask_m)
    else:
        tl.atomic_add(o_ptrs, acc.squeeze(1), mask=mask_m)

def _bgmv_shrink(input, lora, lora_indices, scaling, out=None):
    batch, N, K = input.shape
    assert lora.shape[1:] == (N, K), f"Incompatible shapes: input {input.shape}, lora {lora.shape}"
    assert lora_indices.shape == (batch,), f"Incompatible shapes: lora_indices {lora_indices.shape}, expected ({batch},)"
    
    if out is None:
        out = torch.empty((batch, N), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (batch, N), f"Incompatible shapes: out {out.shape}, expected ({batch}, {N})"

    BLOCK_N = 16
    BLOCK_K = 16
    SPLIT_K = 4

    def grid(meta):
        return (triton.cdiv(N, BLOCK_N) * SPLIT_K * batch,)

    _bgmv_shrink_kernel[grid](
        input, lora, out,
        N, K,
        lora_indices,
        scaling,
        input.stride(0), input.stride(2),
        lora.stride(0), lora.stride(2),
        out.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K
    )

    return out
