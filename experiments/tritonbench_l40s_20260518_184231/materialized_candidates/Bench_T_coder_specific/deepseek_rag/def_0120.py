import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel_g64(inputs_ptr, qw_ptr, mn_ptr, 
                    scale_ptr, output_ptr,
                    IC: tl.constexpr, OC: tl.constexpr, bit: tl.constexpr, 
                    OC_PER_PH: tl.constexpr, PACK_FACTOR: tl.constexpr, BLOCK_SIZE):
    """
    Computes GEMV (group_size = 64).

    Args:
    inputs: vector of shape [batch_size, IC];
    qw: matrix of shape [OC, IC / 8];
    output: vector of shape [OC];
    mn: matrix of shape [OC, NG];
    scale: matrix of shape [OC, NG];

    Notes:
    One cannot infer group_size from the shape of scaling factors.
    the second dimension is rounded up to a multiple of PACK_FACTOR.
    """
    group_size = 64
    oc_idx = tl.program_id(axis=0) * OC_PER_PH + tl.arange(0, OC_PER_PH)
    batch_idx = tl.program_id(axis=1)
    num_groups = IC // group_size
    num_groups_packed = tl.cdiv(num_groups, PACK_FACTOR)
    weight_w = IC // PACK_FACTOR
    num = 0xFF >> (8-bit)
    accumulator = tl.zeros((OC_PER_PH,), dtype=tl.float32)
    for group_idx in range(0, num_groups):
        scale = tl.load(scale_ptr + oc_idx[:, None] * num_groups + group_idx)
        mn = tl.load(mn_ptr + oc_idx[:, None] * num_groups + group_idx)
        cur_qw_ptr = qw_ptr + oc_idx[:, None] * weight_w + group_idx * (64 // PACK_FACTOR) + tl.arange(0, 64 // PACK_FACTOR)[None, :]
        qw = tl.load(cur_qw_ptr)
        for i in range(PACK_FACTOR):
            w_fp = qw & num
            w_fp = w_fp * scale + mn
            qw = qw >> bit
            cur_inp_ptr = inputs_ptr + batch_idx * IC + group_idx * 64 + i + tl.arange(0, 64 // PACK_FACTOR)[None, :] * PACK_FACTOR
            cur_input = tl.load(cur_inp_ptr)
            accumulator += tl.sum(cur_input * w_fp, 1)
    ptr = output_ptr + oc_idx + batch_idx * OC
    tl.store(ptr, accumulator)

def gemv_fwd(bit, group_size, inp, qweight, mn, scale):
    B, IC = inp.shape
    OC = qweight.shape[0]
    BLOCK_SIZE = 32
    OC_PER_PH = 32
    PACK_FACTOR = 32 // bit
    assert group_size == 64
    output = torch.empty((B, OC), device=inp.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(OC, META['OC_PER_PH']), B
    )
    gemv_kernel_g64[grid](inp, qweight, mn, scale, output, 
                       IC, OC, bit, OC_PER_PH, PACK_FACTOR, BLOCK_SIZE)
    return output

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor: 
    # Check if A is a matrix and x, y are vectors
    assert A.ndim == 2 and x.ndim == 1 and y.ndim == 1
    # Compute the matrix-vector product
    y[:] = alpha * torch.mv(A, x) + beta * y
    # Compute the dot product
    result = torch.dot(y, x)
    return result
