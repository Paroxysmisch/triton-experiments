import torch
import triton
import triton.language as tl

@triton.jit
def fifth_order_fwd(
    x_ptr, y_ptr, z_ptr,
    out_ptr,
    N,               # total number of points
    OUT_STRIDE,      # leading dimension of out (number of harmonics)
    BLOCK_SIZE: tl.constexpr
):
    # program index
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)

    # loop over elements in this program's block
    for idx in range(BLOCK_SIZE):
        i = start + idx
        if i >= end:
            break

        # load coords
        x = tl.load(x_ptr + i)
        y = tl.load(y_ptr + i)
        z = tl.load(z_ptr + i)

        # compute "fifth-order" real spherical harmonic components (11 outputs here as example)
        # These are illustrative polynomials in x, y, z
        sh0  = x**5
        sh1  = x**4 * y
        sh2  = x**3 * y**2
        sh3  = x**2 * y**3
        sh4  = x * y**4
        sh5  = y**5
        sh6  = x**4 * z
        sh7  = x**3 * y * z
        sh8  = x**2 * y**2 * z
        sh9  = x * y**3 * z
        sh10 = y**4 * z

        # store to output
        tl.store(out_ptr + i*OUT_STRIDE + 0,  sh0)
        tl.store(out_ptr + i*OUT_STRIDE + 1,  sh1)
        tl.store(out_ptr + i*OUT_STRIDE + 2,  sh2)
        tl.store(out_ptr + i*OUT_STRIDE + 3,  sh3)
        tl.store(out_ptr + i*OUT_STRIDE + 4,  sh4)
        tl.store(out_ptr + i*OUT_STRIDE + 5,  sh5)
        tl.store(out_ptr + i*OUT_STRIDE + 6,  sh6)
        tl.store(out_ptr + i*OUT_STRIDE + 7,  sh7)
        tl.store(out_ptr + i*OUT_STRIDE + 8,  sh8)
        tl.store(out_ptr + i*OUT_STRIDE + 9,  sh9)
        tl.store(out_ptr + i*OUT_STRIDE + 10, sh10)


@triton.jit
def fifth_order_bwd(
    x_ptr, y_ptr, z_ptr,
    grad_out_ptr,
    grad_x_ptr, grad_y_ptr, grad_z_ptr,
    N,               # total number of points
    OUT_STRIDE,      # leading dimension of out
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)

    for idx in range(BLOCK_SIZE):
        i = start + idx
        if i >= end:
            break

        # load coords
        x = tl.load(x_ptr + i)
        y = tl.load(y_ptr + i)
        z = tl.load(z_ptr + i)

        # load grad outputs of each harmonic
        g0  = tl.load(grad_out_ptr + i*OUT_STRIDE + 0)
        g1  = tl.load(grad_out_ptr + i*OUT_STRIDE + 1)
        g2  = tl.load(grad_out_ptr + i*OUT_STRIDE + 2)
        g3  = tl.load(grad_out_ptr + i*OUT_STRIDE + 3)
        g4  = tl.load(grad_out_ptr + i*OUT_STRIDE + 4)
        g5  = tl.load(grad_out_ptr + i*OUT_STRIDE + 5)
        g6  = tl.load(grad_out_ptr + i*OUT_STRIDE + 6)
        g7  = tl.load(grad_out_ptr + i*OUT_STRIDE + 7)
        g8  = tl.load(grad_out_ptr + i*OUT_STRIDE + 8)
        g9  = tl.load(grad_out_ptr + i*OUT_STRIDE + 9)
        g10 = tl.load(grad_out_ptr + i*OUT_STRIDE + 10)

        # partial derivatives of each harmonic w.r.t x, y, z
        # sh0 = x^5
        dsh0_dx  = 5.0 * x**4
        dsh0_dy  = 0.0
        dsh0_dz  = 0.0

        # sh1 = x^4 y
        dsh1_dx  = 4.0 * x**3 * y
        dsh1_dy  = x**4
        dsh1_dz  = 0.0

        # sh2 = x^3 y^2
        dsh2_dx  = 3.0 * x**2 * y**2
        dsh2_dy  = 2.0 * x**3 *
