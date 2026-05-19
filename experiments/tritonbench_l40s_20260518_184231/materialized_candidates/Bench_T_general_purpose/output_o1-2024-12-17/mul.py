import triton
import triton.language as tl
import math

# Utility function for type promotion (simplistic example, extend as needed)
def _promote_types(dtype1, dtype2):
    # Order of preference for demonstration
    type_priority = {
        'int32': 1,
        'float32': 2,
        'complex64': 3
    }
    # Default to float32 if not found in map, or choose higher priority type
    p1 = type_priority.get(str(dtype1), 2)
    p2 = type_priority.get(str(dtype2), 2)
    return ['int32', 'float32', 'complex64'][max(p1, p2) - 1]

# Broadcast shape calculation
def _broadcast_shape(shape1, shape2):
    # Reverse shapes for ease of iteration
    rshape1 = shape1[::-1]
    rshape2 = shape2[::-1]
    out = []
    for i in range(max(len(rshape1), len(rshape2))):
        dim1 = rshape1[i] if i < len(rshape1) else 1
        dim2 = rshape2[i] if i < len(rshape2) else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcastable")
        out.append(max(dim1, dim2))
    return tuple(out[::-1])

@triton.jit
def _mul_kernel(
    input_ptr, other_ptr, out_ptr,
    input_stride, other_stride, out_stride,
    n_elements,
    # Meta
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_elem = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0)
    other_elem = tl.load(other_ptr + offsets * other_stride, mask=mask, other=0)
    out_val = input_elem * other_elem

    tl.store(out_ptr + offsets * out_stride, out_val, mask=mask)

def mul(input, other, *, out=None):
    """
    mul(input, other, *, out=None) -> Tensor

    Multiplies 'input' by 'other', supporting broadcasting and type promotion.
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to multiply 'input' by.
    out (Tensor, optional): destination tensor.
    """
    # Convert 'other' to a tensor-like object if it's a scalar
    if not hasattr(other, 'shape'):
        # Assume we have a 'from_scalar' utility or wrap in a tensor
        # For demonstration, mimic a trivial shape (1,) or (1,)* for broadcast
        other_shape = (1,)
        other_data = [other]
        other_dtype = type(other).__name__
    else:
        other_shape = other.shape
        other_data = other._data  # hypothetical raw data access
        other_dtype = other.dtype

    input_shape = input.shape
    input_data = input._data  # hypothetical raw data access
    input_dtype = input.dtype

    # Determine broadcast shape
    out_shape = _broadcast_shape(input_shape, other_shape)

    # Type promotion
    prom_dtype = _promote_types(input_dtype, other_dtype)

    # If out is None, allocate a new tensor
    if out is None:
        # Hypothetical constructor for a new tensor
        out = type(input).zeros(out_shape, dtype=prom_dtype)
    else:
        # Check shape
        if out.shape != out_shape:
            raise ValueError("Output tensor shape does not match broadcast shape.")
        # Check dtype
        if out.dtype != prom_dtype:
            raise ValueError("Output tensor dtype does not match promoted dtype.")

    # Flatten for simplicity
    input_flat = input_data  # assume already flattened or treat as flattened
    other_flat = other_data
    out_flat = out._data

    n_elements = 1
    for s in out_shape:
        n_elements *= s

    # Compute broadcast strides for kernel
    # If broadcast dim is 1, we keep stride = 0 for that array
    # else stride = 1
    def make_stride(shape1, bshape):
        # returns either 0 or 1 for each element if broadcast
        if len(shape1) == 0:
            return 0
        stride = []
        # simplistic assumption: shape is flattened
        size1 = 1
        for d in shape1:
            size1 *= d
        size_b = 1
        for d in bshape:
            size_b *= d
        if size1 == 1 and size_b > 1:
            return 0
        return 1

    in_stride = make_stride(input_shape, out_shape)
    other_stride = make_stride(other_shape, out_shape)
    out_stride = 1  # writing each element distinctly

    # Grid
    BLOCK_SIZE = 1024
    grid = (math.ceil(n_elements / BLOCK_SIZE),)

    _mul_kernel[grid](
        input_flat, other_flat, out_flat,
        in_stride, other_stride, out_stride,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
