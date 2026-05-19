import triton
import triton.language as tl

@triton.jit
def fifth_order_fwd(
    x_ptr,  # Pointer to input coordinates (x, y, z)
    y_ptr,  # Pointer to output spherical harmonics
    n_elements,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets * 3 + 0, mask=mask)
    y = tl.load(x_ptr + offsets * 3 + 1, mask=mask)
    z = tl.load(x_ptr + offsets * 3 + 2, mask=mask)

    r = tl.sqrt(x * x + y * y + z * z)
    theta = tl.acos(z / r)
    phi = tl.atan2(y, x)

    # Compute fifth-order spherical harmonics
    Y50 = 0.125 * tl.sqrt(11 / (2 * tl.pi)) * (35 * tl.cos(theta)**5 - 30 * tl.cos(theta)**3 + 3 * tl.cos(theta))
    Y51 = 0.125 * tl.sqrt(21 / tl.pi) * tl.sin(theta) * (7 * tl.cos(theta)**4 - 8 * tl.cos(theta)**2 + 1) * tl.cos(phi)
    Y52 = 0.125 * tl.sqrt(21 / (2 * tl.pi)) * tl.sin(theta)**2 * (7 * tl.cos(theta)**3 - 3 * tl.cos(theta)) * tl.cos(2 * phi)
    Y53 = 0.125 * tl.sqrt(7 / (2 * tl.pi)) * tl.sin(theta)**3 * (7 * tl.cos(theta)**2 - 1) * tl.cos(3 * phi)
    Y54 = 0.125 * tl.sqrt(7 / (2 * tl.pi)) * tl.sin(theta)**4 * (5 * tl.cos(theta)) * tl.cos(4 * phi)
    Y55 = 0.125 * tl.sqrt(7 / (2 * tl.pi)) * tl.sin(theta)**5 * tl.cos(5 * phi)

    # Store results
    tl.store(y_ptr + offsets * 6 + 0, Y50, mask=mask)
    tl.store(y_ptr + offsets * 6 + 1, Y51, mask=mask)
    tl.store(y_ptr + offsets * 6 + 2, Y52, mask=mask)
    tl.store(y_ptr + offsets * 6 + 3, Y53, mask=mask)
    tl.store(y_ptr + offsets * 6 + 4, Y54, mask=mask)
    tl.store(y_ptr + offsets * 6 + 5, Y55, mask=mask)
