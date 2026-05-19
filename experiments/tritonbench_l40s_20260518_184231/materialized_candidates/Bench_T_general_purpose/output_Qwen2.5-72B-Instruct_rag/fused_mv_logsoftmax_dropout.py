import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr, dropout_mask_ptr, dropout_prob, training, n, m, seed, stride_input_n, stride_input_m, stride_vec_m, stride_output_n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = batch_start + BLOCK_SIZE

    # Matrix-vector multiplication
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(m):
        input_chunk = tl.load(input_ptr + batch_start * stride_input_n + i * stride_input_m, mask=batch_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
        vec_chunk = tl.load(vec_ptr + i * stride_vec_m)
        z += input_chunk * vec_chunk

    # Log-softmax
    z_max = tl.max(z, axis=0)
    z = z - z_max
    exp_z = tl.exp(z)
    sum_exp_z = tl.sum(exp_z, axis=0)
    log_softmax = z - tl.log(sum_exp_z)

    # Dropout
    if training:
        dropout_mask = tl.rand(seed, pid) < (1 - dropout_prob)
        log_softmax = tl.where(dropout_mask, log_softmax / (1 - dropout_prob), -float('inf'))

    # Store the result
    tl.store(output_ptr + batch_start * stride_output_n, log_softmax, mask=batch_start + tl.arange(0, BLOCK_SIZE) < n)
    if training:
        tl.store(dropout_mask_ptr + batch_start * stride_output_n, dropout_mask, mask=batch_start + tl.arange(0, BLOCK_SIZE) < n)

import torch
import triton
import triton.language as tl

@triton.heuristics({
    'BLOCK_SIZE': lambda *args, **meta: triton.next_power_of_2(args[0].shape[0]),
})
@triton.jit
def fused_mv_logsoftmax_dropout(
    input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None
):
    assert dim in [0, -1], "dim must be 0 or -1"
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec must be compatible for matrix-vector multiplication"

    n, m = input.shape
    if out is None:
        out = torch.empty_like(input) if inplace else torch.empty((n,), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n,), "out must have shape (n,)"

    dropout_mask = torch.empty((n,), dtype=torch.bool, device=input.device) if training else None

    grid = (triton.cdiv(n, 1024),)
    fused_mv_logsoftmax_dropout_kernel[grid](
        input, vec, out, dropout_mask, p, training, n, m, torch.randint(0, 2**32, (1,), device=input.device).item(),
        input.stride(0), input.stride(1), vec.stride(0), out.stride(0), 1024
    )

    return out
