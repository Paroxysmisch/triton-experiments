import triton
import triton.language as tl

# Define the kernel for converting float8 to float16
@triton.jit
def kernel_f8_to_f16(
    X: tl.tensor,  # Input tensor of type int8 (interpreted as float8)
    Y: tl.tensor,  # Output tensor of type float16
    BLOCK_SIZE: int,
):
    pid = tl.program_id(axis=0)
    num_elements = X.shape[0]
    offsets = pid * BLOCK_SIZE
    Xs = tl.load(X + offsets, mask=offsets < num_elements, other=0.0)
    Ys = tl.bitcast(Xs, tl.float16)
    tl.store(Y + offsets, Ys, mask=offsets < num_elements)

# Define the kernel for converting float16 or float32 to float8
@triton.jit
def kernel_f16_to_f8(
    X: tl.tensor,  # Input tensor of type float16 or float32
    Y: tl.tensor,  # Output tensor of type int8 (interpreted as float8)
    BLOCK_SIZE: int,
):
    pid = tl.program_id(axis=0)
    num_elements = X.shape[0]
    offsets = pid * BLOCK_SIZE
    Xs = tl.load(X + offsets, mask=offsets < num_elements, other=0.0)
    Ys = tl.bitcast(Xs, tl.int8)
    tl.store(Y + offsets, Ys, mask=offsets < num_elements)
