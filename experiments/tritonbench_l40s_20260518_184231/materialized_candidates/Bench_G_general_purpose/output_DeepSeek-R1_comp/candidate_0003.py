import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    max_seq_len, head_dim, scale,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    q_offset = batch_idx * stride_q_b + head_idx * stride_q_h + seq_idx * stride_q_s
    q = tl.load(Q + q_offset + tl.arange(0, head_dim), mask=seq_idx < max_seq_len, other=0.0)

    running_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    running_sum = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for k_block in range(0, max_seq_len, BLOCK):
        k_range = k_block + tl.arange(0, BLOCK)
        k_mask = k_range < max_seq_len

        k_offsets = batch_idx * stride_k_b + head_idx * stride_k_h + k_range * stride_k_s
        k = tl.load(K + k_offsets[:, None] + tl.arange(0, head_dim)[None, :], mask=k_mask[:, None], other=0.0)

        qk = tl.sum(q[None, :] * k, axis=1) * scale
        qk = tl.where(k_mask, qk, float('-inf'))

        curr_max = tl.max(qk, axis=0)
        new_max = tl.maximum(running_max, curr_max)

        exp_old = tl.exp(running_max - new_max)
        exp_curr = tl.exp(qk - new_max)

        running_sum = running_sum * exp_old + tl.sum(exp_curr, axis=0)
        running_max = new_max

        v_offsets = batch_idx * stride_v_b + head_idx * stride_v_h + k_range * stride_v_s
        v = tl.load(V + v_offsets[:, None] + tl.arange(0, head_dim)[None, :], mask=k_mask[:, None], other=0.0)

        acc = acc * exp_old + tl.sum(exp_curr[:, None] * v, axis=0)

    acc = acc / running_sum
    out_offset = batch_idx * stride_out_b + head_idx * stride_out_h + seq_idx * stride_out_s
    tl.store(Out + out_offset + tl.arange(0, head_dim), acc, mask=seq_idx < max_seq_len)

def context_attention_fwd(q, k, v, output):
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)

    BLOCK = 128
    if head_dim <= 64:
        BLOCK = 64
    if 'T4' in torch.cuda.get_device_name(0):
        BLOCK = 64  # Optimize for T4's memory constraints

    grid = (batch_size, num_heads, seq_len)
    
    _fwd_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len, head_dim, scale,
        BLOCK=BLOCK,
    )

# Example usage
if __name__ == "__main__":
    B, H, S, D = 4, 8, 1024, 128
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = torch.empty_like(q)

    context_attention_fwd(q, k, v, output)
    torch.cuda.synchronize()
