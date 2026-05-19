import triton
import triton.language as tl

# Example Tensor class for demonstration purposes.
# In a real scenario, this could be integrated with a framework's Tensor class.
class Tensor:
    def __init__(self, data, shape=None):
        self.data = data
        # If shape is not provided, assume it's a 1D array
        self.shape = shape if shape is not None else (len(data),)

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

def _broadcast_shapes(shape1, shape2):
    """
    Broadcast two shapes according to broadcasting rules.
    """
    # Reverse for easier right-to-left handling
    rev_shp1 = list(shape1[::-1])
    rev_shp2 = list(shape2[::-1])
    out = []
    for i in range(max(len(rev_shp1), len(rev_shp2))):
        dim1 = rev_shp1[i] if i < len(rev_shp1) else 1
        dim2 = rev_shp2[i] if i < len(rev_shp2) else 1
        if dim1 != 1 and dim2 != 1 and dim1 != dim2:
            raise ValueError("Shapes are not broadcastable.")
        out.append(max(dim1, dim2))
    return tuple(out[::-1])

def _expand_to_shape(tensor, new_shape):
    """
    Expand tensor data to match new_shape for broadcasting.
    For simplicity, this will create a new flattened list in broadcasted form.
    """
    # If tensor already matches the new shape, return it directly
    if tensor.shape == new_shape:
        return tensor

    # Otherwise, expand
    new_numel = 1
    for s in new_shape:
        new_numel *= s

    # Flatten old data
    old_data = tensor.data
    old_shape = tensor.shape
    old_numel = tensor.numel()

    # Prepare broadcasted output
    new_data = []
    # We'll iterate over the new shape in a naive manner to replicate data
    for idx in range(new_numel):
        # Convert idx into indices for the new shape
        strides = []
        tmp = idx
        for s in reversed(new_shape):
            strides.append(tmp % s)
            tmp //= s
        strides.reverse()
        # Now map to old shape indices
        old_indices = []
        for i, s in enumerate(old_shape):
            if s == 1:
                old_indices.append(0)  # broadcast dimension
            else:
                old_indices.append(strides[i])
        # Convert old_indices to 1D offset
        offset = 0
        mul = 1
        for s in reversed(old_shape):
            offset += old_indices[-1] * mul
            old_indices.pop()
            mul *= s
        new_data.append(old_data[offset % old_numel])  # safeguard if needed

    return Tensor(new_data, new_shape)

@triton.jit
def _pow_kernel(
    x_ptr, exponent_ptr, out_ptr,
    n_elements, exponent_is_scalar, scalar_val,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if exponent_is_scalar:
        e = scalar_val
    else:
        e = tl.load(exponent_ptr + offsets, mask=mask)
    result = x ** e
    tl.store(out_ptr + offsets, result, mask=mask)

def pow(input: Tensor, exponent, *, out=None) -> Tensor:
    """
    pow(input, exponent, *, out=None) -> Tensor

    If exponent is a scalar:
        out[i] = input[i]^exponent
    If exponent is a tensor:
        out[i] = input[i]^exponent[i]
    Broadcasts shapes if needed.
    """
    if not isinstance(input, Tensor):
        raise TypeError("input must be a Tensor")
    if isinstance(exponent, Tensor):
        # Broadcast shapes
        out_shape = _broadcast_shapes(input.shape, exponent.shape)
        x_broadcasted = _expand_to_shape(input, out_shape)
        e_broadcasted = _expand_to_shape(exponent, out_shape)
        x_data = x_broadcasted.data
        e_data = e_broadcasted.data
        n_elements = len(x_data)
        if out is None:
            out = Tensor([0.0] * n_elements, out_shape)
        out_data = out.data

        # Launch kernel
        grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
        _pow_kernel[grid]( 
            x_ptr=x_data,
            exponent_ptr=e_data,
            out_ptr=out_data,
            n_elements=n_elements,
            exponent_is_scalar=False,
            scalar_val=0.0,  # not used
            BLOCK_SIZE=1024
        )
        out.shape = out_shape
        return out

    else:
        # exponent is scalar
        if not isinstance(exponent, (float, int)):
            raise TypeError("exponent must be a float, int, or Tensor")
        x_data = input.data
        n_elements = input.numel()
        if out is None:
            out = Tensor([0.0] * n_elements, input.shape)
        out_data = out.data

        # Launch kernel
        grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
        _pow_kernel[grid](
            x_ptr=x_data,
            exponent_ptr=out_data,  # not used for load
            out_ptr=out_data,
            n_elements=n_elements,
            exponent_is_scalar=True,
            scalar_val=float(exponent),
            BLOCK_SIZE=1024
        )
        out.shape = input.shape
        return out
