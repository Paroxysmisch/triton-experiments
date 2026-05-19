import torch

def torch.permute_copy(input, dims) -> Tensor:
    # Ensure the input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a Tensor")

    # Ensure dims is a tuple
    if not isinstance(dims, tuple):
        raise TypeError("Dims must be a tuple")

    # Ensure all elements in dims are integers
    if not all(isinstance(dim, int) for dim in dims):
        raise TypeError("All elements in dims must be integers")

    # Ensure all elements in dims are unique
    if len(dims) != len(set(dims)):
        raise ValueError("All elements in dims must be unique")

    # Ensure all elements in dims are within the range [-input.ndim, input.ndim-1]
    if any(dim < -input.ndim or dim > input.ndim - 1 for dim in dims):
        raise IndexError("All elements in dims must be within the range [-input.ndim, input.ndim-1]")

    # Call the Triton kernel
    output = _permute_copy(input, dims)

    # Return the output tensor
    return output
