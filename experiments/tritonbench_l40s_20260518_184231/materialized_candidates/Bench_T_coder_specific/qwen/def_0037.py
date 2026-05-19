triton
import triton
import triton.language as tl

@triton.jit
def fused_cosine_embedding_loss_with_normalization_kernel(
    input1_ptr,
    input2_ptr,
    target_ptr,
    output_ptr,
    n,
    d,
    margin,
    stride_n_input1,
    stride_d_input1,
    stride_n_input2,
    stride_d_input2,
    stride_n_target,
    stride_n_output,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    x_start = min(block_start + tl.arange(0, BLOCK_SIZE), n)
    input1_x = input1_ptr + x_start * stride_n_input1
    input2_x = input2_ptr + x_start * stride_n_input2
    target_x = target_ptr + x_start * stride_n_target
    output_x = output_ptr + x_start * stride_n_output

    # Load data into shared memory
    input1_shared = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    input2_shared = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    target_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(input1_shared, tl.load(input1_x), mask=x_start < n)
    tl.store(input2_shared, tl.load(input2_x), mask=x_start < n)
    tl.store(target_shared, tl.load(target_x), mask=x_start < n)

    # Compute norms
    norm_input1 = tl.sum(tl.square(input1_shared), axis=1)
    norm_input2 = tl.sum(tl.square(input2_shared), axis=1)
    norm_input1 = tl.maximum(norm_input1, 1e-8)  # Avoid division by zero
    norm_input2 = tl.maximum(norm_input2, 1e-8)
    input1_normalized = input1_shared / tl.sqrt(norm_input1[:, None])
    input2_normalized = input2_shared / tl.sqrt(norm_input2[:, None])

    # Compute cosine similarity
    cos_sim = tl.dot(input1_normalized, input2_normalized.T)

    # Compute loss
    loss = tl.where(
        target_shared == 1,
        -cos_sim,
        tl.maximum(cos_sim - margin, 0)
    )

    # Reduce loss
    if reduction == 'mean':
        loss = tl.mean(loss, axis=0)
    elif reduction == 'sum':
        loss = tl.sum(loss, axis=0)

    # Store result
    tl.store(output_x, loss[0], mask=True)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
    ],
    key=['n', 'd']
)
def fused_cosine_embedding_loss_with_normalization(
    input1: tl.tensor,
    input2: tl.tensor,
    target: tl.tensor,
    margin: float = 0,
    reduction: str = 'mean',
    n: int,
    d: int,
):
    output = tl.zeros((n,), dtype=tl.float32)
    fused_cosine_embedding_loss_with_normalization_kernel[
        grid=n // 128 + 1,
        block=(128,)
    ](
        input1.data_ptr(),
        input2.data_ptr(),
        target.data_ptr(),
        output.data_ptr(),
        n,
        d,
        margin,
        input1.stride(0),
        input1.stride(1),
        input2.stride(0),
        input2.stride(1),
        target.stride(0),
        output.stride(0),
        BLOCK_SIZE=128
    )
    return output


# Example usage:
input1 = tl.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tl.float32)
input2 = tl.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=tl.float32)
target = tl.tensor([1, -1], dtype=tl.int32)
margin = 0.5
reduction = 'mean'
result = fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin, reduction)
print(result)
