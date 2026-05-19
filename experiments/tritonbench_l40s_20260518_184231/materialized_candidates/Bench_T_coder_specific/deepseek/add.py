import triton
import triton.language as tl

@triton.jit
def add(out_ptr, in_ptr, other_ptr, other_val, alpha):
    # Get the indices
    pid = tl.program_id(axis=0)
    bid = tl.block_id(axis=0)
    tid = tl.thread_id(axis=0)

    # Compute the global id
    gid = pid * tl.num_threads(axis=0) + tid

    # Load the input and other values
    in_val = tl.load(in_ptr + gid)
    other_val = tl.load(other_ptr + gid) if other_ptr is not None else other_val

    # Perform the operation
    out_val = in_val + alpha * (other_val if other_ptr is None else tl.load(other_ptr + gid))

    # Store the result
    tl.store(out_ptr + gid, out_val)

def add_wrapper(input, other, alpha=1, out=None):
    # Get the pointers to the input and output tensors
    in_ptr = triton.pointers.allocate_shared(input)
    out_ptr = triton.pointers.allocate_shared(out) if out is not None else in_ptr

    # Get the pointer to the other tensor or value
    other_ptr = triton.pointers.allocate_shared(other) if isinstance(other, triton.Tensor) else None

    # Get the other value
    other_val = other if not isinstance(other, triton.Tensor) else 0

    # Call the kernel
    add[1, 1, 1](out_ptr, in_ptr, other_ptr, other_val, alpha)

    # Return the output tensor
    return out
