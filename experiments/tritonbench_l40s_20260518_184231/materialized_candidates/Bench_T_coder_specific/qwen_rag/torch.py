import triton
import triton.language as tl

@triton.jit
def permute_kernel(
    A_ptr,          # Pointer to the input tensor
    O_ptr,          # Pointer to the output tensor
    A_shape,        # Shape of the input tensor (as a tuple)
    A_strides,      # Strides of the input tensor (as a tuple)
    O_strides,      # Strides of the output tensor (as a tuple)
    N,              # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr = 32
):
    # Determine the flattened index for the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = idx % N
    
    # Unpack the original indices from the flattened index
    orig_indices = []
    for i in range(len(A_shape)):
        orig_indices.append(idx // A_strides[i])
        idx %= A_strides[i]
    
    # Apply the permutation to get the new indices
    new_indices = [orig_indices[dims[i]] for i in range(len(dims))]
    
    # Flatten the new indices to get the output index
    output_idx = sum(new_indices[i] * O_strides[i] for i in range(len(dims)))
    
    # Load the value from the input tensor and store it in the output tensor
    value = tl.load(A_ptr + idx)
    tl.store(O_ptr + output_idx, value)
