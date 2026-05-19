import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    X_ptr,  # Pointer to the input tensor (batched vectors)
    W_ptr,  # Pointer to the sparse matrix (CSR format)
    V_ptr,  # Pointer to the LoRA weights
    Y_ptr,  # Pointer to the output tensor
    stride_xb,  # Stride of X in the batch dimension
    stride_xc,  # Stride of X in the feature dimension
    stride_wb,  # Stride of W in the batch dimension
    stride_wi,  # Stride of W in the row index dimension
    stride_wj,  # Stride of W in the column index dimension
    stride_wv,  # Stride of W in the value dimension
    stride_vb,  # Stride of V in the batch dimension
    stride_vc,  # Stride of V in the feature dimension
    stride_yb,  # Stride of Y in the batch dimension
    stride_yc,  # Stride of Y in the feature dimension
    B,  # Batch size
    M,  # Number of rows in the sparse matrix
    N,  # Number of columns in the sparse matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Get the program ID
    batch_id = pid // (M // BLOCK_SIZE)  # Determine the batch ID
    row_id = (pid % (M // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Determine the row ID

    # Load the input vector for the current batch
    X = tl.load(X_ptr + batch_id * stride_xb + row_id * stride_xc, mask=row_id < M, other=0.0)

    # Initialize the output vector
    Y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load the sparse matrix and LoRA weights
    for i in range(0, N, BLOCK_SIZE):
        col_id = i + tl.arange(0, BLOCK_SIZE)
        mask = (row_id < M) & (col_id < N)

        # Load the values from the sparse matrix
        W_values = tl.load(W_ptr + batch_id * stride_wb + row_id * stride_wi + col_id * stride_wj + col_id * stride_wv, mask=mask, other=0.0)

        # Load the LoRA weights
        V_values = tl.load(V_ptr + batch_id * stride_vb + col_id * stride_vc, mask=mask, other=0.0)

        # Perform the sparse matrix-vector multiplication
        Y += X * W_values * V_values

    # Store the result in the output tensor
    tl.store(Y_ptr + batch_id * stride_yb + row_id * stride_yc, Y, mask=row_id < M)

import torch
import triton
import triton.language as tl

def _sgmv_expand_slice(X, W, V, Y, block_size=128):
    # Get the dimensions of the input tensors
    B, M = X.shape
    N = V.shape[1]

    # Determine the grid and block sizes
    grid = (B * (M // block_size),)

    # Launch the Triton kernel
    _sgmv_expand_slice_kernel[grid](
        X,  # Input tensor
        W,  # Sparse matrix (CSR format)
        V,  # LoRA weights
        Y,  # Output tensor
        X.stride(0),  # Stride of X in the batch dimension
        X.stride(1),  # Stride of X in the feature dimension
        W.stride(0),  # Stride of W in the batch dimension
        W.stride(1),  # Stride of W in the row index dimension
        W.stride(2),  # Stride of W in the column index dimension
        W.stride(3),  # Stride of W in the value dimension
        V.stride(0),  # Stride of V in the batch dimension
        V.stride(1),  # Stride of V in the feature dimension
        Y.stride(0),  # Stride of Y in the batch dimension
        Y.stride(1),  # Stride of Y in the feature dimension
        B,  # Batch size
        M,  # Number of rows in the sparse matrix
        N,  # Number of columns in the sparse matrix
        BLOCK_SIZE=block_size,  # Block size for parallelism
    )
