import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    pid_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    dq_ptr,
    dk_ptr,
    BACKWARD_PASS: tl.constexpr,
):
    # Get unique id for each program instance
    pid = tl.program_id(0)

    # Get row and column indices
    row = tl.program_id(1)
    col = tl.program_id(2)

    # Get stride
    stride = tl.program_id(0) * tl.cuda.warp_size()

    # Load cos and sin arrays
    cos = cos_ptr[pid][row]
    sin = sin_ptr[pid][row]

    # Load query and key matrices
    q = q_ptr[pid][row][col]
    k = k_ptr[pid][row][col]

    # Apply rotary position embeddings
    if BACKWARD_PASS:
        q_new = q * cos - k * sin
        k_new = q * sin + k * cos
    else:
        q_new = q * cos + k * sin
        k_new = q * sin - k * cos

    # Store new query and key matrices
    q_ptr[pid][row][col] = q_new
    k_ptr[pid][row][col] = k_new

    # Load gradients
    dq = dq_ptr[pid][row][col]
    dk = dk_ptr[pid][row][col]

    # Apply rotary position embeddings to gradients
    if BACKWARD_PASS:
        dq_new = dq * cos - dk * sin
        dk_new = dq * sin + dk * cos
    else:
        dq_new = dq * cos + dk * sin
        dk_new = dq * sin - dk * cos

    # Store new gradients
    dq_ptr[pid][row][col] = dq_new
    dk_ptr[pid][row][col] = dk_new

def rope_backward(
    q,
    k,
    dq,
    dk,
    cos,
    sin,
    BACKWARD_PASS: bool = False,
):
    # Pad inputs to power-of-two dimensions
    q = tl.pad(q, (0, 2**tl.log2(q.shape[0]) - q.shape[0]))
    k = tl.pad(k, (0, 2**tl.log2(k.shape[0]) - k.shape[0]))
    dq = tl.pad(dq, (0, 2**tl.log2(dq.shape[0]) - dq.shape[0]))
    dk = tl.pad(dk, (0, 2**tl.log2(dk.shape[0]) - dk.shape[0]))

    # Trigger kernel
    _triton_rope[q.shape[0], q.shape[1], q.shape[2]](
        tl.cta_id(0),
        q,
        k,
        cos,
        sin,
        dq,
        dk,
        BACKWARD_PASS,
    )

    # Return gradients
    return dq, dk
