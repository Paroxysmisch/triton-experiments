@triton.jit
def pow_kernel(input_ptr, exponent, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the index for the current program/thread
    pid = tl.program_id(0)
    # Compute the starting pointer for the current block
    block_start_ptr = input_ptr + pid * BLOCK_SIZE
    # Generate offsets within the block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the pointers for the current block
    pointers = block_start_ptr + offsets
    # Load the block from global memory into SRAM with masking for out-of-bounds
    block = tl.load(pointers, mask=offsets < n_elements)
    # Perform the pow operation
    result = block ** exponent
    # Store the result back to global memory
    tl.store(pointers, result, mask=offsets < n_elements)
