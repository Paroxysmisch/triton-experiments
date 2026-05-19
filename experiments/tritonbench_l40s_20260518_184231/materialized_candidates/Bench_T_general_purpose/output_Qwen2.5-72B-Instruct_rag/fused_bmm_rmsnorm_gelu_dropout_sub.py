import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    output_ptr, input1_ptr, input2_ptr, other_ptr, weight_ptr, stride1, stride2, stride3, stride4, stride5, 
    B, N, M, P, dropout_p, training, eps, seed, BLOCK_SIZE: tl.constexpr
):
    # Compute the batch matrix multiplication
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    if row_idx >= B or col_idx >= N:
        return

    # Initialize pointers
    input1_row_start_ptr = input1_ptr + row_idx * stride1 + col_idx * stride2
    input2_col_start_ptr = input2_ptr + row_idx * stride3
    output_row_start_ptr = output_ptr + row_idx * stride4 + col_idx * stride5
    other_row_start_ptr = other_ptr + row_idx * stride4 + col_idx * stride5

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for m in range(0, M, BLOCK_SIZE):
        cols = m + tl.arange(0, BLOCK_SIZE)
        mask = cols < M
        input1_vals = tl.load(input1_row_start_ptr + cols * stride2, mask=mask, other=0.0)
        input2_vals = tl.load(input2_col_start_ptr + cols * stride3, mask=mask, other=0.0)
        acc += input1_vals * input2_vals

    # Store the result of the batch matrix multiplication
    tl.store(output_row_start_ptr, acc, mask=tl.arange(0, BLOCK_SIZE) < P)

    # Perform RMS normalization
    square_sum = tl.sum(acc * acc, axis=0)
    rms = tl.sqrt(square_sum / P + eps)
    normed = acc / rms

    # Apply GELU activation
    if approximate == 'none':
        normed = 0.5 * normed * (1.0 + tl.math.erf(normed / tl.sqrt(2.0)))
    elif approximate == 'tanh':
        normed = normed * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (normed + 0.044715 * normed * normed * normed)))

    # Apply dropout
    if training:
        rng = tl.rand(seed, row_idx * N + col_idx)
        mask = rng > dropout_p
        normed = tl.where(mask, normed, 0.0)
        normed /= (1.0 - dropout_p)

    # Subtract the other tensor
    other_vals = tl.load(other_row_start_ptr, mask=tl.arange(0, BLOCK_SIZE) < P, other=0.0)
    result = normed - other_vals

    # Store the final result
    tl.store(output_row_start_ptr, result, mask=tl.arange(0, BLOCK_SIZE) < P)

import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    # Ensure the shapes are compatible
    assert input1.shape == (B, N, M), f"input1 shape must be (B, N, M), got {input1.shape}"
    assert input2.shape == (B, M, P), f"input2 shape must be (B, M, P), got {input2.shape}"
    assert other.shape == (B, N, P) or other.shape == (1, N, P) or other.shape == (B, 1, P) or other.shape == (1, 1, P), f"other shape must be broadcastable to (B, N, P), got {other.shape}"

    # Ensure the normalized shape is correct
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    assert normalized_shape == [P], f"normalized_shape must be [P], got {normalized_shape}"

    # Ensure the approximate parameter is valid
    assert approximate in ['none', 'tanh'], f"approximate must be 'none' or 'tanh', got {approximate}"

    # Seed for dropout
    seed = torch.randint(0, 2**32, (1,), device=input1.device).item()

    # Call the Triton kernel
    grid = (B, N)
    block = (P, 1, 1)
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid, block](
        out, input1, input2, other, torch.ones(P, device=input1.device), 
        input1.stride(0), input1.stride(1), input2.stride(0), input2.stride(1), 
        B, N, M, P, dropout_p, training, eps, seed, BLOCK_SIZE=P
    )

    return out

import torch

# Test case
B, N, M, P = 2, 3, 4, 5
input1 = torch.randn(B, N, M, device='cuda')
input2 = torch.randn(B, M, P, device='cuda')
other = torch.randn(B, N, P, device='cuda')

# Triton implementation
output_triton = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape=P, dropout_p=0.5, training=True, approximate='none', eps=1e-5)

# PyTorch implementation
output_pytorch = torch.bmm(input1, input2)
output_pytorch = torch.nn.functional.layer_norm(output_pytorch, (P,), eps=1e-5)
output_pytorch = torch.nn.functional.gelu(output_pytorch)
output_pytorch = torch.nn.functional.dropout(output_pytorch, p=0.5, training=True)
output_pytorch = output_pytorch - other

# Compare the results
print("Triton output:", output_triton)
print("PyTorch output:", output_pytorch)
print("Difference:", torch.abs(output_triton - output_pytorch).max())
