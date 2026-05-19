import triton
import triton.language as tl

@triton.jit
def tensordot_kernel(
    a_ptr, b_ptr, out_ptr,
    a_shape, b_shape, out_shape,
    a_strides, b_strides, out_strides,
    contract_dims_a, contract_dims_b,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Compute the index for the output tensor
    idx = tl.arange(0, BLOCK_SIZE)
    idx_out = pid * BLOCK_SIZE + idx

    # Calculate the position in the output tensor
    out_pos = tl.zeros([len(out_shape)], dtype=tl.int32)
    for i in range(len(out_shape)):
        out_pos[i] = (idx_out // out_strides[i]) % out_shape[i]

    # Initialize the output element
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the contracting dimensions
    for k in range(contract_dims_a[0], contract_dims_a[1]):
        a_idx = tl.zeros([len(a_shape)], dtype=tl.int32)
        b_idx = tl.zeros([len(b_shape)], dtype=tl.int32)

        # Compute the indices for a and b
        for i in range(len(a_shape)):
            if i in contract_dims_a:
                a_idx[i] = k
            else:
                a_idx[i] = out_pos[i]

        for j in range(len(b_shape)):
            if j in contract_dims_b:
                b_idx[j] = k
            else:
                b_idx[j] = out_pos[j]

        # Load the elements from a and b
        a_val = tl.load(a_ptr + tl.dot(a_idx, a_strides))
        b_val = tl.load(b_ptr + tl.dot(b_idx, b_strides))

        # Perform the multiplication and accumulation
        acc += a_val * b_val

    # Store the result in the output tensor
    tl.store(out_ptr + idx_out, acc)

import torch
from typing import Union, Tuple, List

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Determine the dimensions to contract
    if isinstance(dims, int):
        contract_dims_a = list(range(-dims, 0))
        contract_dims_b = list(range(dims))
    elif isinstance(dims, (tuple, list)) and len(dims) == 2:
        contract_dims_a, contract_dims_b = dims
    else:
        raise ValueError("Invalid dims argument")

    # Validate dimensions
    if len(contract_dims_a) != len(contract_dims_b):
        raise ValueError("Mismatch in the number of dimensions to contract")

    # Calculate the output shape
    out_shape = list(a.shape[:-len(contract_dims_a)]) + list(b.shape[len(contract_dims_b):])

    # Flatten the tensors
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Calculate strides
    a_strides = list(a.stride())
    b_strides = list(b.stride())
    out_strides = list(torch.empty(out_shape).stride())

    # Prepare pointers
    a_ptr = a_flat.data_ptr()
    b_ptr = b_flat.data_ptr()
    out_ptr = torch.empty(out_shape, device=a.device, dtype=a.dtype).data_ptr()

    # Launch the kernel
    BLOCK_SIZE = 1024  # Define a block size
    grid = (torch.prod(torch.tensor(out_shape)) + BLOCK_SIZE - 1) // BLOCK_SIZE

    tensordot_kernel[grid](
        a_ptr, b_ptr, out_ptr,
        a.shape, b.shape, out_shape,
        a_strides, b_strides, out_strides,
        contract_dims_a, contract_dims_b,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape the output tensor
    out_tensor = torch.tensor(out_ptr).reshape(out_shape)

    return out_tensor
