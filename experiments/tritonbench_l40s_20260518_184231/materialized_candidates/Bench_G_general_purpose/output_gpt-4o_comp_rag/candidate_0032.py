The code you've provided implements a custom Triton kernel to compute the softmax function over batched and multi-headed logic data. Let's go through the implementation details of both the Triton kernel `_fwd_kernel_token_softmax` and its wrapper function `token_softmax_fwd`.

### Triton Kernel: `_fwd_kernel_token_softmax`

This kernel is designed to compute the softmax probabilities for each batch and head independently. It leverages Triton's parallel computing capabilities to process each logic row efficiently.

#### Parameters:
- **Logics**: The input tensor containing logits.
- **B_Start_Loc**: The starting indices of each sequence in the batch.
- **B_Seqlen**: The lengths of each sequence.
- **Prob_Out**: The output tensor where the computed softmax probabilities will be stored.
- **stride_logic_h, stride_logic_bs, stride_prob_h, stride_prob_bs**: Memory strides for accessing elements in `Logics` and `Prob_Out`.
- **BLOCK_SIZE**: A constant that defines the maximum block size for processing.

#### Process:
1. **Batch and Head Identification**: Uses `tl.program_id` to determine which batch and head are being processed.
2. **Column Offsets and Sequence Information**: Calculates column offsets and loads the sequence length and start index for the current batch.
3. **Load Logits**: Loads the logits for the current head and batch, using a mask to handle sequences of different lengths.
4. **Softmax Computation**:
   - Subtracts the maximum value in the row for numerical stability.
   - Computes the exponentials of the adjusted logits.
   - Normalizes by dividing by the sum of the exponentials.
5. **Store Results**: Stores the computed softmax probabilities in the output tensor.

### Wrapper Function: `token_softmax_fwd`

This function is responsible for setting up and launching the Triton kernel.

#### Parameters:
- **Logics**: Input tensor of logits.
- **B_Start_Loc**: Starting indices of each sequence.
- **B_Seqlen**: Lengths of each sequence.
- **Prob_Out**: Output tensor for softmax probabilities.
- **max_input_len**: The maximum input length across all sequences.

#### Process:
1. **Determine BLOCK_SIZE**: Uses `triton.next_power_of_2` to determine the appropriate block size based on `max_input_len`.
2. **Calculate Number of Warps**: Sets the number of warps based on the block size to optimize performance.
3. **Launch Kernel**: Invokes the Triton kernel `_fwd_kernel_token_softmax` with the calculated configurations for batch and head dimensions.

### Key Points:
- **Numerical Stability**: The softmax computation is performed in a numerically stable manner by subtracting the maximum value from each element before exponentiation.
- **Parallel Execution**: The use of Triton's parallel computing capabilities allows for efficient computation across batches and heads.
- **No Gradient Computation**: The `@torch.no_grad()` decorator indicates that this function is intended for inference and does not compute gradients.

This implementation is well-suited for applications where softmax probabilities are needed for batched sequences, such as in attention mechanisms in neural networks.
