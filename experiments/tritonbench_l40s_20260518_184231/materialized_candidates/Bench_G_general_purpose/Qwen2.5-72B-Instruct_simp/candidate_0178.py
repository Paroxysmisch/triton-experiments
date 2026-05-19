import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(
    input_ptr,  # *input_ptr: pointer to input tensor of shape (B, T, D)
    A_ptr,      # *A_ptr: pointer to diagonal state transition matrix of shape (B, D, D)
    B_ptr,      # *B_ptr: pointer to input matrix of shape (B, D, 1)
    C_ptr,      # *C_ptr: pointer to output matrix of shape (B, 1, D)
    output_ptr, # *output_ptr: pointer to output tensor of shape (B, T, 1)
    B,          # batch size
    T,          # sequence length
    D,          # state dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // T
    tid = pid % T

    if bid >= B or tid >= T:
        return

    input_offset = bid * T * D + tid * D
    output_offset = bid * T + tid
    A_offset = bid * D * D
    B_offset = bid * D
    C_offset = bid * D

    state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(tid, T):
        input_vec = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE))
        A_vec = tl.load(A_ptr + A_offset + tl.arange(0, BLOCK_SIZE))
        B_vec = tl.load(B_ptr + B_offset + tl.arange(0, BLOCK_SIZE))
        C_vec = tl.load(C_ptr + C_offset + tl.arange(0, BLOCK_SIZE))

        state = state * A_vec + input_vec * B_vec
        output = tl.dot(C_vec, state)

        tl.store(output_ptr + output_offset, output)
        input_offset += D
        output_offset += 1

@triton.jit
def diag_ssm_forward_kernel_complex(
    input_ptr,  # *input_ptr: pointer to input tensor of shape (B, T, D)
    A_ptr,      # *A_ptr: pointer to diagonal state transition matrix of shape (B, D, D)
    B_ptr,      # *B_ptr: pointer to input matrix of shape (B, D, 1)
    C_ptr,      # *C_ptr: pointer to output matrix of shape (B, 1, D)
    output_ptr, # *output_ptr: pointer to output tensor of shape (B, T, 1)
    B,          # batch size
    T,          # sequence length
    D,          # state dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // T
    tid = pid % T

    if bid >= B or tid >= T:
        return

    input_offset = bid * T * D + tid * D
    output_offset = bid * T + tid
    A_offset = bid * D * D
    B_offset = bid * D
    C_offset = bid * D

    state_real = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    state_imag = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(tid, T):
        input_real = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE))
        input_imag = tl.load(input_ptr + input_offset + D + tl.arange(0, BLOCK_SIZE))
        A_real = tl.load(A_ptr + A_offset + tl.arange(0, BLOCK_SIZE))
        A_imag = tl.load(A_ptr + A_offset + D + tl.arange(0, BLOCK_SIZE))
        B_real = tl.load(B_ptr + B_offset + tl.arange(0, BLOCK_SIZE))
        B_imag = tl.load(B_ptr + B_offset + D + tl.arange(0, BLOCK_SIZE))
        C_real = tl.load(C_ptr + C_offset + tl.arange(0, BLOCK_SIZE))
        C_imag = tl.load(C_ptr + C_offset + D + tl.arange(0, BLOCK_SIZE))

        state_real, state_imag = (
            state_real * A_real - state_imag * A_imag + input_real * B_real - input_imag * B_imag,
            state_real * A_imag + state_imag * A_real + input_real * B_imag + input_imag * B_real
        )
        output_real = tl.dot(C_real, state_real) - tl.dot(C_imag, state_imag)
        output_imag = tl.dot(C_real, state_imag) + tl.dot(C_imag, state_real)

        tl.store(output_ptr + output_offset, output_real)
        tl.store(output_ptr + output_offset + D, output_imag)
        input_offset += D
        output_offset += 1

@triton.jit
def diag_ssm_backward_kernel(
    grad_output_ptr,  # *grad_output_ptr: pointer to gradient of output tensor of shape (B, T, 1)
    A_ptr,            # *A_ptr: pointer to diagonal state transition matrix of shape (B, D, D)
    B_ptr,            # *B_ptr: pointer to input matrix of shape (B, D, 1)
    C_ptr,            # *C_ptr: pointer to output matrix of shape (B, 1, D)
    grad_input_ptr,   # *grad_input_ptr: pointer to gradient of input tensor of shape (B, T, D)
    B,                # batch size
    T,                # sequence length
    D,                # state dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // T
    tid = pid % T

    if bid >= B or tid >= T:
        return

    grad_output_offset = bid * T + tid
    grad_input_offset = bid * T * D + tid * D
    A_offset = bid * D * D
    B_offset = bid * D
    C_offset = bid * D

    grad_state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(tid, -1, -1):
        grad_output = tl.load(grad_output_ptr + grad_output_offset)
        C_vec = tl.load(C_ptr + C_offset + tl.arange(0, BLOCK_SIZE))
        A_vec = tl.load(A_ptr + A_offset + tl.arange(0, BLOCK_SIZE))
        B_vec = tl.load(B_ptr + B_offset + tl.arange(0, BLOCK_SIZE))

        grad_state = grad_state * A_vec + grad_output * C_vec
        grad_input = grad_state * B_vec

        tl.store(grad_input_ptr + grad_input_offset, grad_input)
        grad_input_offset -= D
        grad_output_offset -= 1

@triton.jit
def diag_ssm_backward_kernel_complex(
    grad_output_ptr,  # *grad_output_ptr: pointer to gradient of output tensor of shape (B, T, 1)
    A_ptr,            # *A_ptr: pointer to diagonal state transition matrix of shape (B, D, D)
    B_ptr,            # *B_ptr: pointer to input matrix of shape (B, D, 1)
    C_ptr,            # *C_ptr: pointer to output matrix of shape (B, 1, D)
    grad_input_ptr,   # *grad_input_ptr: pointer to gradient of input tensor of shape (B, T, D)
    B,                # batch size
    T,                # sequence length
    D,                # state dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // T
    tid = pid % T

    if bid >= B or tid >= T:
        return

    grad_output_offset = bid * T + tid
    grad_input_offset = bid * T * D + tid * D
    A_offset = bid * D * D
    B_offset = bid * D
    C_offset = bid * D

    grad_state_real = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    grad_state_imag = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for t in range(tid, -1, -1):
        grad_output_real = tl.load(grad_output_ptr + grad_output_offset)
        grad_output_imag = tl.load(grad_output_ptr + grad_output_offset + D)
        C_real = tl.load(C_ptr + C_offset + tl.arange(0, BLOCK_SIZE))
        C_imag = tl.load(C_ptr + C_offset + D + tl.arange(0, BLOCK_SIZE))
        A_real = tl.load(A_ptr + A_offset + tl.arange(0, BLOCK_SIZE))
        A_imag = tl.load(A_ptr + A_offset + D + tl.arange(0, BLOCK_SIZE))
        B_real = tl.load(B_ptr + B_offset + tl.arange(0, BLOCK_SIZE))
        B_imag = tl.load(B_ptr + B_offset + D + tl.arange(0, BLOCK_SIZE))

        grad_state_real, grad_state_imag = (
            grad_state_real * A_real - grad_state_imag * A_imag + grad_output_real * C_real - grad_output_imag * C_imag,
            grad_state_real * A_imag + grad_state_imag *
