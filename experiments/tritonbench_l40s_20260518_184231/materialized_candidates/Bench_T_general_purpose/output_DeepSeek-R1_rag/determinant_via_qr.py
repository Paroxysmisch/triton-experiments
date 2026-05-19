import torch
import triton
import triton.language as tl

@triton.jit
def prod_diagonal_kernel(output_ptr, input_ptr, input_row_stride, input_col_stride, n, is_complex: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < n

    element_stride = input_row_stride + input_col_stride
    input_ptrs = input_ptr + off * element_stride

    real_product = 1.0
    imag_product = 0.0

    if is_complex:
        real_ptrs = input_ptrs * 2
        imag_ptrs = real_ptrs + 1
        reals = tl.load(real_ptrs, mask=mask, other=1.0)
        imags = tl.load(imag_ptrs, mask=mask, other=0.0)
        for i in range(BLOCK_SIZE):
            if off[i] < n:
                current_real = reals[i]
                current_imag = imags[i]
                new_real = real_product * current_real - imag_product * current_imag
                new_imag = real_product * current_imag + imag_product * current_real
                real_product = new_real
                imag_product = new_imag
    else:
        elements = tl.load(input_ptrs, mask=mask, other=1.0)
        for i in range(BLOCK_SIZE):
            if off[i] < n:
                real_product *= elements[i]

    if tl.program_id(0) == 0:
        if is_complex:
            tl.store(output_ptr, real_product)
            tl.store(output_ptr + 1, imag_product)
        else:
            tl.store(output_ptr, real_product)

def determinant_via_qr(A, *, mode='reduced', out=None):
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    n = A.size(0)
    Q, R = torch.qr(A, mode=mode)
    det_Q = torch.det(Q)
    
    is_complex = A.is_complex()
    if is_complex:
        prod_r_diag = torch.empty(2, device=A.device, dtype=torch.float32)
    else:
        prod_r_diag = torch.empty(1, device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE = triton.next_power_of_two(n)
    grid = (1,)
    prod_diagonal_kernel[grid](
        prod_r_diag,
        R,
        R.stride(0),
        R.stride(1),
        n,
        is_complex,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if is_complex:
        prod_r_diag_complex = torch.view_as_complex(prod_r_diag.unsqueeze(0)).squeeze()
    else:
        prod_r_diag_complex = prod_r_diag.squeeze()
    
    determinant = det_Q * prod_r_diag_complex
    
    if out is not None:
        out.copy_(determinant)
        return out
    return determinant
