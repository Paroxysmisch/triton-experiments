import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr,
    n, m,
    p, training,
    input_row_stride, input_col_stride,
    vec_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    SEED: tl.constexpr,
):
    # Compute matrix-vector product for each row and store in shared memory
    pid = tl.program_id(axis=0)
    row = pid
    if row >= n:
        return

    # Compute dot product for the row
    acc = tl.zeros(tl.float32, (1,))
    for col in range(0, m, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < m
        a = tl.load(input_ptr + row * input_row_stride + cols * input_col_stride, mask=mask, other=0.0)
        v = tl.load(vec_ptr + cols * vec_stride, mask=mask, other=0.0)
        acc += tl.sum(a * v)
    z = acc

    # Compute max across all elements using a single block reduction
    max_z = tl.max(z, axis=0)
    # Compute sum of exp(z - max_z)
    z_stable = z - max_z
    exp_z = tl.exp(z_stable)
    sum_exp = tl.sum(exp_z, axis=0)
    log_sum_exp = tl.log(sum_exp)
    log_softmax = z_stable - log_sum_exp

    # Apply dropout
    if training:
        # Generate random mask
        random = tl.rand(SEED + row)
        mask = random > p
        scale = 1.0 / (1.0 - p)
        output = tl.where(mask, log_softmax * scale, 0.0)
    else:
        output = log_softmax

    tl.store(output_ptr + row * output_row_stride, output)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    assert input.dim() == 2, "Input must be a 2D matrix"
    assert vec.dim() == 1, "vec must be a 1D vector"
    assert input.size(1) == vec.size(0), "Input columns must match vec size"
    assert dim in (0, -1), "dim must be 0 or -1 for log_softmax on 1D tensor"

    n, m = input.shape
    device = input.device

    if inplace:
        assert not input.is_cuda, "Inplace operation is not supported for CUDA tensors in this kernel"
        output = input.select(1, 0)  # Placeholder, adjust according to actual logic
    else:
        output = torch.empty(n, device=device, dtype=input.dtype)

    if out is not None:
        assert out.shape == output.shape, "out tensor has incorrect shape"
        output = out

    BLOCK_SIZE = 128
    grid = (n,)
    seed = torch.randint(0, 2**32, (1,), device=device).item()

    _fused_mv_logsoftmax_dropout_kernel[grid](
        input, vec, output,
        n, m,
        p, training,
        input.stride(0), input.stride(1),
        vec.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        SEED=seed,
    )

    return output
