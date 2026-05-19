import torch
import triton
import triton.language as tl


@triton.jit
def fused_mlp_softmax_dropout_layer_norm_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    residual_ptr,
    output_ptr,
    M,
    N,
    D,
    K,
    n_cols_block_size,
    BLOCK_SIZE: tl.constexpr,
    n_group: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    RMS_NORM_EPS: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    SAVE_INPUT: tl.constexpr,
):
    """
    Arguments:
    1. input_ptr: pointer to the input, shape [M, N, D] -- cannot be strided in 'N' dimension!
               It is expected to be contiguous in 'N', but it can have arbitrary strides in
               leading dimensions.
    2. weight1_ptr: pointer to the first linear layer weights, shape [D, K]
    3. weight2_ptr: pointer to the second linear layer weights, shape [K, D]
    4. residual_ptr: pointer to the residual, shape [M, N, D] -- cannot be strided in 'N' dimension!
               It is expected to be contiguous in 'N', but it can have arbitrary strides in
               leading dimensions.
    5. output_ptr: pointer to the output, shape [M, N, D].
               It is expected to be contiguous in 'N', but it can have arbitrary strides in
               leading dimensions.
    6. M: precomputed M (number of blocks in 'input'/'residual' in the first dimension)
    7. N: precomputed N (number of elements in the second dimension)
    8. D: precomputed D (number of elements in the third dimension)
    9. K: hidden dimension (linear layer size)
    10. n_cols_block_size: size of the blocks that the kernel uses to iterate over K.
    11. BLOCK_SIZE: block size. AKA "vector size". The size of the vector that will be
                processed by a single instantiation of the innermost loop.
                Must be a power-of-two.
    12. n_group: autotune param. The groups to split the main loop into.
                Reduces pressure on L1 cache and allows for better instruction level
                parallelism.
    13. IS_RMS_NORM: whether to use RMS norm or regular layer norm
    14. RMS_NORM_EPS: epsilon passed into RMS norm
    15. DROPOUT_P: probability that an element of the softmax output is dropped.
                    If 0, dropout is disabled.
    16. RECOMPUTE_OUTPUT: if set, the kernel re-computes and stores the output.
                    If unset, the kernel loads the output from `output_ptr`.
    17. SAVE_INPUT: if set, the kernel saves the intermediate result of the computation
                    (i.e. output of linear1, softmax, dropout) into `output_ptr`.
                    Use this for debugging or when you plan to run the kernel a second time
                    with the same `output_ptr` but different `input_ptr`.
                    In the latter case, make sure to pass `RECOMPUTE_OUTPUT=True` in the
                    second call.
    """

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # -----------------------------------------------------------
    # Step 1: Block multiplication
    # We want to compute:
    #   Z = X @ W1 @ W2
    # Where:
    #   X is of shape [M, N, D]
    #   W1 is of shape [D, K]
    #   W2 is of shape [K, D]
    #   Z is of shape [M, N, D]
    #   M = X.shape[0]
    #   N = X.shape[1]
    #   K = W1.shape[1]
    #   D = X.shape[-1] == W1.shape[0] == W2.shape[-1]

    # Get the address of all the elements that we want to load in our blocks
    # We will store these addresses in three vectors of length BLOCK_SIZE.
    # Note that we are loading X in blocks of size BLOCK_SIZE, which means
    # that BLOCK_SIZE should divide the number of elements in X's leading
    # dimension.

    # offsets for the address of the first element of each block
    input_block_offsets = (
        pid_m * N * D + pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    )

    # To save memory, we put all the offsets in a single tensor
    # Shape: [BLOCK_SIZE]
    input_block_ptrs = (
        input_ptr + input_block_offsets
        if SAVE_INPUT
        else input_ptr + input_block_offsets
    )

    # Unpack ptrs into BLOCK_SIZE separate ptrs
    # This increases register pressure but reduces the number of loops we need
    # to write significantly
    input_chunk_ptrs = tl.broadcast_to(
        input_block_ptrs[:, None], (BLOCK_SIZE, n_cols_block_size)
    ).to(tl.pointer_type(tl.float32))

    # Similar to above but for W1
    weight1_block_ptrs = (
        weight1_ptr
        + tl.arange(0, n_cols_block_size)[:, None] * D
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )

    # And for W2
    weight2_block_ptrs = (
        weight2_ptr
        + tl.arange(0, n_cols_block_size)[:, None] * D
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )

    # Now let's do the actual multiplication
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over K in blocks of size BLOCK_SIZE
    for k in range(0, tl.cdiv(K, n_cols_block_size)):
        # Fetch a chunk of X
        x = tl.load(
            input_chunk_ptrs,
            mask=(k * n_cols_block_size + tl.arange(0, n_cols_block_size)) < K,
            other=0.0,
        )

        # Move closer to getting correct dims for Gemm
        x = tl.reshape(x, (BLOCK_SIZE, n_cols_block_size))
        # Treat it as COLUMN MAJOR!!!
        x = tl.trans(x)

        # Fetch a chunk of W1
        w1 = tl.load(
            weight1_block_ptrs.to(tl.pointer_type(tl.float32)),
            mask=(k * n_cols_block_size + tl.arange(0, n_cols_block_size)) < K,
            other=0.0,
        )

        # First matmul: X @ W1
        acc += tl.dot(x, w1)

        # Update ptrs to next chunk of data
        input_chunk_ptrs += n_cols_block_size
        weight1_block_ptrs += n_cols_block_size * D

    # At this point, we have acc == X @ W1.

    # Now we apply softmax
    # Softmax is just a scaled logsumexp
    # As in the PyTorch library, we compute logits-clipped-to-never-exp as:
    #   logits_clipped = max(logits - large_number, 0)
    # Then we compute
    #   numerator = exp(logits_clipped)
    #   denominator = sum(exp(logits_clipped))
    # Then finally:
    #   softmax_output = numerator / denominator

    # Subtract maximum for numerical stability
    acc_minus_max = acc - tl.max(acc, axis=1)[:, None]
    # Our "large number"
    big_constant = tl.log(1.0000001e+5)
    acc_clipped = tl.where(acc_minus_max < big_constant, acc_minus_max, big_constant)
    # Numerator: exp(logits - max)
    numerator = tl.exp(acc_clipped)
    # Denominator: sum(exp(logits - max))
    denominator = tl.sum(numerator, axis=1)[:, None]

    softmax_output = numerator / denominator

    # Apply Dropout
    if DROPOUT_P != 0.0:
        keep_mask = tl.rand(tl.float32, (BLOCK_SIZE, BLOCK_SIZE)) > DROPOUT_P
        softmax_output = tl.where(keep_mask, softmax_output / (1 - DROPOUT_P), 0.0)

    if RECOMPUTE_OUTPUT:
        # Store the result for later use
        output_block_ptrs = (
            output_ptr + input_block_offsets
        )

        output_chunk_ptrs = tl.broadcast_to(
            output_block_ptrs[:, None], (BLOCK_SIZE, n_cols_block_size)
        ).to(tl.pointer_type(tl.float32))

        weight2_trans_block_ptrs = (
            weight2_ptr
            + tl.arange(0, n_cols_block_size)[:, None] * D
            + tl.arange(0, BLOCK_SIZE)[None, :]
        )

        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, n_cols_block_size)):
            w2 = tl.load(
                weight2_trans_block_ptrs.to(tl.pointer_type(tl.float32)),
                mask=(k * n_cols_block_size + tl.arange(0, n_cols_block_size)) < K,
                other=0.0,
            )
            weight2_trans_block_ptrs += n_cols
