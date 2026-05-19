import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 1024  # You can adjust this based on your GPU and problem size
NUM_WARPS = 4      # Number of warps for parallel execution

@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the indices for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input elements
    a = tl.load(a_ptr + offsets, mask=offsets < n_elements, other=0.0)
    b = tl.load(b_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute SiLU activation
    silu_a = a * tl.sigmoid(a)

    # Compute the element-wise product
    c = silu_a * b

    # Store the result
    tl.store(c_ptr + offsets, c, mask=offsets < n_elements)

def swiglu_forward(a, b):
    assert a.shape == b.shape, "Input tensors must have the same shape"
    n_elements = a.numel()
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    c = torch.empty_like(a)

    _swiglu_forward_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS
    )

    return c

@triton.jit
def _swiglu_backward_kernel(a_ptr, b_ptr, dc_ptr, da_ptr, db_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the indices for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input elements and output gradient
    a = tl.load(a_ptr + offsets, mask=offsets < n_elements, other=0.0)
    b = tl.load(b_ptr + offsets, mask=offsets < n_elements, other=0.0)
    dc = tl.load(dc_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute SiLU and its derivative
    sigmoid_a = tl.sigmoid(a)
    silu_a = a * sigmoid_a
    d_silu_a = sigmoid_a * (1 + a * (1 - sigmoid_a))

    # Compute gradients
    da = dc * b * d_silu_a
    db = dc * silu_a

    # Store the gradients
    tl.store(da_ptr + offsets, da, mask=offsets < n_elements)
    tl.store(db_ptr + offsets, db, mask=offsets < n_elements)

def swiglu_backward(a, b, dc):
    assert a.shape == b.shape == dc.shape, "All tensors must have the same shape"
    n_elements = a.numel()
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    da = torch.empty_like(a)
    db = torch.empty_like(b)

    _swiglu_backward_kernel[grid](
        a_ptr=a, b_ptr=b, dc_ptr=dc, da_ptr=da, db_ptr=db,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS
    )

    return da, db
