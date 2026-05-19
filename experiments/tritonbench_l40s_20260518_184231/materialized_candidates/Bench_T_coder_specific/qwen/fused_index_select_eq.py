import torch
from typing import Optional

def fused_index_select_eq(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    other: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    assert input.dim() > dim >= 0, "Invalid dim"
    assert index.dim() == 1, "Index must be a 1D tensor"
    assert index.dtype in [torch.int32, torch.int64], "Index must be of type IntTensor or LongTensor"

    # Prepare strides
    x_shape = list(input.shape)
    index_shape = list(index.shape)
    y_shape = list(other.shape)
    x_stride = [input.stride(d) for d in range(input.dim())]
    index_stride = [index.stride(d) for d in range(index.dim())]
    y_stride = [other.stride(d) for d in range(other.dim())]
    o_stride = [out.stride(d) for d in range(out.dim())]

    # Flatten the tensors to simplify indexing
    flat_x = input.view(-1)
    flat_index = index.view(-1)
    flat_other = other.view(-1)
    flat_out = out.view(-1)

    # Number of elements to process
    num_elements = flat_x.numel()

    # Launch the Triton kernel
    fused_index_select_eq_launcher(
        flat_x.data_ptr(),
        flat_index.data_ptr(),
        flat_other.data_ptr(),
        flat_out.data_ptr(),
        x_shape,
        index_shape,
        y_shape,
        x_stride,
        index_stride,
        y_stride,
        o_stride,
        num_elements,
        dim
    )

    return out
