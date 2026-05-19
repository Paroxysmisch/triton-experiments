import triton
import triton.language as tl

# Forward Kernel
@triton.jit
def _geglu_tanh_forward_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program's starting point
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=offsets < n_elements, other=0.0)
    b = tl.load(b_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute GEGLU using tanh approximation
    gate = tl.sigmoid(b)
    c = a * gate

    # Store the result
    tl.store(c_ptr + offsets, c, mask=offsets < n_elements)

# Backward Kernel
@triton.jit
def _geglu_tanh_backward_kernel(a_ptr, b_ptr, grad_output_ptr, grad_a_ptr, grad_b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program's starting point
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load inputs
    a = tl.load(a_ptr + offsets, mask=offsets < n_elements, other=0.0)
    b = tl.load(b_ptr + offsets, mask=offsets < n_elements, other=0.0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute gate and its gradient
    gate = tl.sigmoid(b)
    grad_gate = gate * (1 - gate)

    # Compute gradients
    grad_a = grad_output * gate
    grad_b = grad_output * a * grad_gate

    # Store the gradients
    tl.store(grad_a_ptr + offsets, grad_a, mask=offsets < n_elements)
    tl.store(grad_b_ptr + offsets, grad_b, mask=offsets < n_elements)

# Helper function to determine settings
def calculate_settings(n_elements):
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

# Wrapper for forward pass
def geglu_forward(a, b):
    assert a.shape == b.shape, "Input tensors must have the same shape"
    n_elements = a.numel()
    c = torch.empty_like(a)

    BLOCK_SIZE, num_warps = calculate_settings(n_elements)

    _geglu_tanh_forward_kernel[(n_elements // BLOCK_SIZE,)](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    return c

# Wrapper for backward pass
def geglu_backward(a, b, grad_output):
    assert a.shape == b.shape == grad_output.shape, "All tensors must have the same shape"
    n_elements = a.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)

    BLOCK_SIZE, num_warps = calculate_settings(n_elements)

    _geglu_tanh_backward_kernel[(n_elements // BLOCK_SIZE,)](a, b, grad_output, grad_a, grad_b, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    return grad_a, grad_b
