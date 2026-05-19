import triton
import triton.language as tl

@triton.jit
def kernel_fma(A_ptr, B_ptr, C_ptr, bias_ptr, activation_type, M, N, K, stride_A, stride_B, stride_C):
    # Define the indices
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # Load the inputs
    A = tl.load(A_ptr + row * stride_A + col)
    B = tl.load(B_ptr + row * stride_B + col)
    C = tl.load(C_ptr + row * stride_C + col)
    bias = tl.load(bias_ptr + row)

    # Compute the dot product
    dot_product = tl.dot(A, B)

    # Apply the activation
    if activation_type == 'tanh':
        activation = tl.tanh(dot_product + bias)
    elif activation_type == 'relu':
        activation = tl.max(dot_product + bias, 0)
    # Add more activation functions as needed

    # Store the result
    tl.store(C_ptr + row * stride_C + col, activation)
