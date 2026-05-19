import triton
import triton.language as tl

# Define constants
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 16
EPS = 1e-5

@triton.jit
def ff_llama_kernel(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    x_shape, w1_shape, w3_shape, rms_w_shape, output_shape,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, EPS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(x_shape[0], BLOCK_SIZE_M)
    grid_n = tl.cdiv(x_shape[1], BLOCK_SIZE_N)
    grid_k = tl.cdiv(x_shape[2], BLOCK_SIZE_K)
    
    row = pid // (grid_n * grid_k) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = (pid % (grid_n * grid_k)) // grid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = (pid % (grid_n * grid_k)) % grid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(0, x_shape[0], BLOCK_SIZE_M):
        x_chunk = tl.load(x_ptr + (i + row[:, None]) * x_shape[1] + col, mask=(i + row[:, None] < x_shape[0])[:, None])
        w1_chunk = tl.load(w1_ptr + (k[:, None]) * w1_shape[1] + col, mask=(k[:, None] < w1_shape[0])[:, None])
        w3_chunk = tl.load(w3_ptr + (k[:, None]) * w3_shape[1] + col, mask=(k[:, None] < w3_shape[0])[:, None])
        rms_w_chunk = tl.load(rms_w_ptr + (k[:, None]) * rms_w_shape[1] + col, mask=(k[:, None] < rms_w_shape[0])[:, None])
        
        acc1 += tl.dot(x_chunk, w1_chunk)
        acc2 += tl.dot(x_chunk, w3_chunk)
        
        # RMS normalization
        rms = tl.sqrt(rms_w_chunk * acc1 * acc1 + EPS)
        acc1 /= rms
    
    # Sigmoid activation
    sigmoid = 1.0 / (1.0 + tl.exp(-acc1))
    
    # Combine accumulators
    output = acc2 * sigmoid
    
    # Store output
    output_ptr += (pid // (grid_n * grid_k) * output_shape[0] + row[:, None]) * output_shape[1] + col
    tl.store(output_ptr, output, mask=(pid // (grid_n * grid_k) * output_shape[0] + row[:, None] < output_shape[0])[:, None])

@triton.jit
def kernel_ff(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    x_shape, w1_shape, w3_shape, rms_w_shape, output_shape,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, EPS: tl.constexpr
):
    # Assert correct types and shapes
    assert x_shape[1] == w1_shape[0] == w3_shape[0]
    assert w1_shape[1] == w3_shape[1]
    assert rms_w_shape[1] == w1_shape[1]
    assert output_shape[0] == x_shape[0] and output_shape[1] == x_shape[1]
    
    # Transpose weight matrices if necessary
    w1_ptr = w1_ptr + (tl.arange(0, w1_shape[1])[:, None] * w1_shape[0] + tl.arange(0, w1_shape[0]))
    w3_ptr = w3_ptr + (tl.arange(0, w3_shape[1])[:, None] * w3_shape[0] + tl.arange(0, w3_shape[0]))
    
    # Set up grid dimensions
    grid = lambda meta: (
        tl.cdiv(x_shape[0], BLOCK_SIZE_M) * tl.cdiv(x_shape[1], BLOCK_SIZE_N) * tl.cdiv(x_shape[2], BLOCK_SIZE_K),
    )
    
    # Invoke the kernel
    ff_llama_kernel[grid](x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
                          x_shape, w1_shape, w3_shape, rms_w_shape, output_shape,
                          BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPS)

# Example usage
x = ...  # Your input tensor
w1 = ...  # Your weight tensor for w1
w3 = ...  # Your weight tensor for w3
rms_w = ...  # Your RMS weight tensor
output = ...  # Your output tensor

kernel_ff(
    x_ptr=x.data_ptr(),
    w1_ptr=w1.data_ptr(),
    w3_ptr=w3.data_ptr(),
    rms_w_ptr=rms_w.data_ptr(),
    output_ptr=output.data_ptr(),
    x_shape=x.shape,
    w1_shape=w1.shape,
    w3_shape=w3.shape,
    rms_w_shape=rms_w.shape,
    output_shape=output.shape,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    EPS=EPS
)
