import triton.language as tl

@triton.jit
def gelu_kernel(x, y, alpha, out):
    # Compute the sum of x and y scaled by alpha
    z = x + alpha * y

    # Compute the GELU function using the error function approximation
    z_gelu = 0.5 * z * (1 + tl.erf(z * 0.7071067811))

    # Write the result to out
    tl.store(out, z_gelu)
