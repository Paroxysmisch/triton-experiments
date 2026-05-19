import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    stride_bh,
    stride_h,
    stride_d,
    stride_bs,
    stride_bsh,
    stride_bsh_x,
    bsh_x,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_step = BLOCK_SIZE_M
    col_start = tl.arange(0, BLOCK_SIZE_N)
    col_step = BLOCK_SIZE_N

    a_ptr = q_ptr + row_start * stride_bh
    b_ptr = k_ptr + col_start * stride_d
    cos_ptr += col_start
    sin_ptr += col_start
    out_a_ptr = out_q_ptr + row_start * stride_bh
    out_b_ptr = out_k_ptr + col_start * stride_d

    for i in range(row_start, row_start + row_step, BLOCK_SIZE_M):
        for j in range(col_start, col_start + col_step, BLOCK_SIZE_N):
            m = i * bsh_x + j
            n = j - i
            if BACKWARD_PASS:
                m, n = n, m
            off_m = tl.max_contiguous(tl.multiple_of(m % stride_bsh, bsh_x), bsh_x)
            off_n = tl.max_contiguous(tl.multiple_of(n, 1), 1)
            cos = tl.load(cos_ptr + off_n, mask=n < stride_bsh_x, other=0.0)
            sin = tl.load(sin_ptr + off_n, mask=n < stride_bsh_x, other=0.0)
            off_a = tl.max_contiguous(tl.multiple_of(m, 2), 2)
            off_b = tl.multiple_of(off_m, 2)
            dim = off_m // 2
            x0 = tl.load(a_ptr + off_a, mask=off_a < stride_bsh, other=0.0)
            x1 = tl.load(a_ptr + off_a + 1, mask=off_a + 1 < stride_bsh, other=0.0)
            y0 = tl.load(b_ptr + off_b, mask=off_b < stride_bsh, other=0.0)
            y1 = tl.load(b_ptr + off_b + 1, mask=off_b + 1 < stride_bsh, other=0.0)
            out_x0 = x0 * cos - y0 * sin
            out_x1 = x1 * cos - y1 * sin
            out_y0 = x0 * sin + y0 * cos
            out_y1 = x1 * sin + y1 * cos
            tl.store(out_a_ptr + off_a, out_x0, mask=off_a < stride_bsh)
            tl.store(out_a_ptr + off_a + 1, out_x1, mask=off_a + 1 < stride_bsh)
            tl.store(out_b_ptr + off_b, out_y0, mask=off_b < stride_bsh)
            tl.store(out_b_ptr + off_b + 1, out_y1, mask=off_b + 1 < stride_bsh)


def rope_forward(q, k, cos, sin, backward_pass=False):
    batch_size, seq_len, head_num, head_dim = q.shape
    assert q.shape == k.shape
    assert cos.shape == sin.shape
    assert cos.shape[0] == sin.shape[0]
    assert cos.shape[1] == seq_len
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    bsh = triton.next_power_of_2(head_num)
    bsh_x = triton.next_power_of_2(head_num + seq_len)
    bsh_m = triton.next_power_of_2(max(head_num, seq_len))
    grid = lambda meta: (triton.cdiv(batch_size * seq_len, meta["BLOCK_SIZE_M"]),)
    num_warps = 4
    _triton_rope[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        bsh_x,
        bsh_m,
        backward_pass,
        BLOCK_SIZE_H=head_dim,
        BLOCK_SIZE_D=head_dim,
        BLOCK_SIZE_M=bsh_m,
        BLOCK_SIZE_N=bsh_m,
        num_warps=num_warps,
        num_stages=1,
    )
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    out_q = out_q.transpose(1, 2)
    out_k = out_k.transpose(1, 2)
    return out_q.contiguous(), out_k.contiguous(), cos, sin
