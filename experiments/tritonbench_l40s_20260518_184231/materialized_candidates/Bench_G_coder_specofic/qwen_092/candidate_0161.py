triton
import triton
import triton.language as tl

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr: tl.tensor,  # Pointer to packed 4-bit floats
    s_ptr: tl.tensor,  # Pointer to scaling factors in e8m0 format
    output_ptr: tl.tensor,  # Pointer where results are stored
    n_elements_in: tl.int32,  # Number of packed elements to process
    # Constants for masks and biases
    SIGN_MASK_F4: tl.constexpr(0x8),
    BIAS_F4: tl.constexpr(0x7),
    SIGN_MASK_E8M0: tl.constexpr(0x80),
    BIAS_E8M0: tl.constexpr(0x7F),
):
    # Define block and grid sizes
    BLOCK_SIZE = 128
    grid_size = tl.cdiv(n_elements_in, BLOCK_SIZE)
    grid = tl.create_program_id(0, grid_size)

    # Each block processes 16 elements (2 packed bytes)
    block_idx = grid * BLOCK_SIZE
    offset = block_idx * 16
    num_elements = min(16, n_elements_in - offset)

    # Load packed bytes
    x_bytes = tl.load(x_ptr + offset // 2, mask=tl.arange(num_elements) < 8)
    s_values = tl.load(s_ptr + offset // 16, mask=tl.arange(num_elements) < 2)

    # Decompose each packed byte into two 4-bit numbers
    x1 = (x_bytes & 0xF0) >> 4
    x2 = x_bytes & 0x0F

    # Convert 4-bit to bf16
    def convert_to_bf16(x, s):
        sign = (x & SIGN_MASK_F4) >> BIAS_F4
        mantissa = x & BIAS_F4
        scale = (s & SIGN_MASK_E8M0) >> BIAS_E8M0
        scale = scale | 0x7F  # Ensure scale is always positive
        result = (sign << 15) | (mantissa << 11) | scale
        return result

    bf16_1 = convert_to_bf16(x1, s_values[0])
    bf16_2 = convert_to_bf16(x2, s_values[1])

    # Store results
    output_ptr[offset] = bf16_1
    output_ptr[offset + 1] = bf16_2
