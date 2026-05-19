The provided Triton code defines a kernel function `rotary_kernel` and a wrapper function `apply_rotary` to perform rotary positional encoding on input matrices using GPUs. Below is a breakdown of how these functions work and how they are implemented:

### `rotary_kernel` Function

The `rotary_kernel` function is a Triton kernel that applies rotary positional encoding to a tensor `X` using precomputed cosine (`COS`) and sine (`SIN`) matrices. The function modifies or populates the output tensor `OUT` with the transformed data. The kernel supports both fixed and variable sequence lengths, controlled by the presence of `CU_SEQLENS`. It also handles interleaved and non-interleaved formats and allows for in-place transformations and conjugate computations if specified.

#### Parameters:
- **Pointers to matrices**: `OUT`, `X`, `COS`, `SIN`, `CU_SEQLENS`, `SEQLEN_OFFSETS`.
- **Matrix dimensions**: `seqlen`, `nheads`, `rotary_dim`, `seqlen_ro`, `CACHE_KEY_SEQLEN`.
- **Strides**: `stride_out_batch`, `stride_out_seqlen`, `stride_out_nheads`, `stride_out_headdim`, `stride_x_batch`, `stride_x_seqlen`, `stride_x_nheads`, `stride_x_headdim`.
- **Meta-parameters**: `BLOCK_K`, `IS_SEQLEN_OFFSETS_TENSOR`, `IS_VARLEN`, `INTERLEAVED`, `CONJUGATE`, `BLOCK_M`.

#### Operation:
1. **Grid Setup**: The kernel operates on a three-dimensional grid, processing batches (`pid_batch`), heads (`pid_head`), and sequences (`pid_m`).
2. **Index Calculation**: Depending on whether variable lengths are used, the indices for accessing `X` and `OUT` are calculated.
3. **Rotary Transformation**: The kernel performs the rotary transformation by loading blocks of data and applying the transformation based on cosine and sine values.
4. **Interleaved Handling**: The kernel handles both interleaved and non-interleaved data formats with conditional handling for conjugation using `CONJUGATE`.
5. **Output Storage**: The transformed data is stored back into the `OUT` tensor.

### `apply_rotary` Function

The `apply_rotary` function is a high-level interface to the Triton kernel. It accepts the input tensor `x`, cosine and sine matrices, sequence length offsets, and optional cumulative sequence lengths (`cu_seqlens`). It determines the execution grid and block sizes, aligning them with the input data shape and configuration.

#### Parameters:
- `x`: Input tensor.
- `cos`: Cosine matrix.
- `sin`: Sine matrix.
- `seqlen_offsets`: Sequence length offsets (int or tensor).
- `cu_seqlens`: Cumulative sequence lengths (optional).
- `max_seqlen`: Maximum sequence length (optional).
- `interleaved`: Flag for interleaved format.
- `inplace`: Flag for in-place transformation.
- `conjugate`: Flag for conjugate computation.

#### Operation:
1. **Shape and Type Checks**: Ensures that the input tensor and matrices have compatible shapes and types.
2. **Output Initialization**: Initializes an output tensor, copying non-rotary parts of `x` if required.
3. **Grid and Block Configuration**: Determines the grid and block sizes for the kernel launch.
4. **Kernel Launch**: Calls the `rotary_kernel` with appropriate arguments, matching the shape and type expectations set within the kernel logic.
5. **Return**: Returns the transformed output tensor.

This design allows for efficient rotary transformations in transformer architectures, leveraging GPU acceleration through Triton.
