import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_stride0, q_stride1, q_stride2,
    k_stride0, k_stride1, k_stride2,
    cos_stride0, cos_stride1,
    sin_stride0, sin_stride1,
    n_head, head_dim, seq_len, batch_size,
    BACKWARD_PASS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // seq_len
    sid = pid % seq_len

    q_offset = bid * q_stride0 + sid * q_stride1
    k_offset = bid * k_stride0 + sid * k_stride1
    cos_offset = sid * cos_stride0
    sin_offset = sid * sin_stride0

    for hid in range(n_head):
        q_head_offset = q_offset + hid * q_stride2
        k_head_offset = k_offset + hid * k_stride2

        q_vec = tl.load(q_ptr + q_head_offset, mask=hid < n_head, other=0.0)
        k_vec = tl.load(k_ptr + k_head_offset, mask=hid < n_head, other=0.0)

        cos_vec = tl.load(cos_ptr + cos_offset + hid * cos_stride1)
        sin_vec = tl.load(sin_ptr + sin_offset + hid * sin_stride1)

        q_rot = q_vec * cos_vec - tl.flip(q_vec, 1) * sin_vec
        k_rot = k_vec * cos_vec - tl.flip(k_vec, 1) * sin_vec

        if BACKWARD_PASS:
            q_rot = q_rot * cos_vec + tl.flip(q_rot, 1) * sin_vec
            k_rot = k_rot * cos_vec + tl.flip(k_rot, 1) * sin_vec

        tl.store(q_ptr + q_head_offset, q_rot, mask=hid < n_head)
        tl.store(k_ptr + k_head_offset, k_rot, mask=hid < n_head)

import triton
import triton.language as tl
import torch

def rope_forward(q, k, cos, sin, n_head, head_dim, seq_len, batch_size):
    # Transpose the query and key matrices to the appropriate format
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()

    # Compute the necessary paddings
    q_pad = triton.util.pad_to_multiple(q, 16, dim=1)
    k_pad = triton.util.pad_to_multiple(k, 16, dim=1)
    cos_pad = triton.util.pad_to_multiple(cos, 16, dim=1)
    sin_pad = triton.util.pad_to_multiple(sin, 16, dim=1)

    # Get the strides
    q_stride0, q_stride1, q_stride2 = q_pad.stride(0), q_pad.stride(1), q_pad.stride(2)
    k_stride0, k_stride1, k_stride2 = k_pad.stride(0), k_pad.stride(1), k_pad.stride(2)
    cos_stride0, cos_stride1 = cos_pad.stride(0), cos_pad.stride(1)
    sin_stride0, sin_stride1 = sin_pad.stride(0), sin_pad.stride(1)

    # Configure the execution grid
    grid = (batch_size * seq_len,)

    # Call the kernel
    _triton_rope[grid](
        q_pad, k_pad, cos_pad, sin_pad,
        q_stride0, q_stride1, q_stride2,
        k_stride0, k_stride1, k_stride2,
        cos_stride0, cos_stride1,
        sin_stride0, sin_stride1,
        n_head, head_dim, seq_len, batch_size,
        BACKWARD_PASS=False
    )

    # Return the matrices to their original shapes
    q = q_pad[:batch_size, :seq_len, :].transpose(0, 1)
    k = k_pad[:batch_size, :seq_len, :].transpose(0, 1)
    cos = cos_pad[:seq_len, :].contiguous()
    sin = sin_pad[:seq_len, :].contiguous()

    return q, k, cos, sin
