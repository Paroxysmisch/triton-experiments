triton
@triton.jit
def _multinomial_sampling_kernel(
    scores: Pointer[Float32],  # Input scores tensor
    indices: Pointer[Int32],    # Output indices tensor
    scores_shape: Int32,        # Shape of the scores tensor (batch_size, num_tokens)
    num_tokens: Int32,          # Number of tokens per batch
    batch_size: Int32,          # Batch size
    seed: Int32,                # Seed for random number generation
    grid_size: i32[2],          # Grid size for parallel execution
    block_size: i32[2],         # Block size for parallel execution
    stride: i32[2]              # Stride for accessing tensor elements
):
    # Calculate the indices for the current thread
    batch_id = triton.program_id(0)
    token_id = triton.program_id(1)
    row = batch_id * stride[0] + token_id
    
    # Initialize random seed and offset
    tid = triton.thread_id(0)
    seed_offset = batch_id * num_tokens + token_id
    random_state = seed + seed_offset
    
    # Initialize cumulative scores and sampled index
    cum_scores = 0.0
    sampled_index = -1
    
    # Iterate over the tokens in the block
    for i in range(BLOCK_N):
        # Generate a random number
        random_value = triton.random.uniform(random_state, low=0.0, high=1.0)
        
        # Update cumulative scores
        cum_scores += scores[row * stride[1] + i]
        
        # Check if the random value falls within the cumulative probability range
        if random_value <= cum_scores:
            sampled_index = i
            break
    
    # Store the sampled index
    indices[row] = sampled_index
