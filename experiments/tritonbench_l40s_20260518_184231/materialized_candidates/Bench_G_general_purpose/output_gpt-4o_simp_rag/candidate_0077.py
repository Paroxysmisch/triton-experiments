The provided code defines two Triton kernels and their respective wrapper functions to perform token attention computations on GPU. Let's break down the functionality of each kernel and wrapper:

### Kernel: `_fwd_kernel_token_att2`

This kernel performs a forward pass of token attention computation. It processes sequences of tokens using batch and head indices to compute output activations. Here's a step-by-step explanation of the kernel:

1. **Kernel Arguments**:
   - `Prob`, `V`, `Out`: Tensors representing probabilities, values, and output respectively.
   - `B_Loc`, `B_Start_Loc`, `B_Seqlen`: Tensors providing location, start location, and sequence length information.
   - `max_input_len`: Maximum length of input sequences.
   - Strides for the tensors to handle data layout.
   - `BLOCK_DMODEL`, `BLOCK_N`: Constants defining block sizes for dimensions.

2. **Kernel Logic**:
   - Identify the current batch and head using `tl.program_id`.
   - Calculate offsets for accessing data blocks.
   - Initialize an accumulator for the weighted sum.
   - Loop over the sequence in blocks (`BLOCK_N` size) and:
     - Load probability values (`p_value`) and value locations (`v_loc`).
     - Load the value tensor (`v_value`) using the locations.
     - Accumulate the weighted sum of the values.
   - Convert the accumulated result to `float16` and store it in the output tensor.

### Wrapper: `token_att_fwd2`

This function sets up and launches the `_fwd_kernel_token_att2` kernel:

1. **Function Arguments**:
   - `prob`, `v`, `out`: Tensors for probabilities, values, and output.
   - `B_Loc`, `B_Start_Loc`, `B_Seqlen`, `max_input_len`: Additional configuration tensors.

2. **Function Logic**:
   - Determine the block size based on Triton version.
   - Set up a grid based on batch and head dimensions.
   - Launch the kernel with appropriate parameters.

### Kernel: `_fwd_kernel_token_att2_int8v`

This kernel extends `_fwd_kernel_token_att2` by adding support for `int8` scaling. It performs a similar computation but includes scaling factors:

1. **Additional Kernel Arguments**:
   - `V_scale`: Tensor for scaling values.
   - Additional strides for the scaling tensor.

2. **Extended Kernel Logic**:
   - Similar to `_fwd_kernel_token_att2`, but includes loading and applying the scaling factors (`vs_value`) to the values.

### Wrapper: `token_att_fwd2_int8v`

This function sets up and launches the `_fwd_kernel_token_att2_int8v` kernel:

1. **Function Arguments**:
   - Similar to `token_att_fwd2`, with an additional `v_scale` tensor.

2. **Function Logic**:
   - Determine the block size based on `max_input_len`.
   - Set up a grid based on batch and head dimensions.
   - Launch the kernel with appropriate parameters.

These kernels and wrappers provide efficient GPU implementations for attention-like operations, leveraging Triton's ability to handle block-level parallelism and customizable kernel execution.
