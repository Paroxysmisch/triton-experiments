import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize weight[K // 8, N], scale[K, N // 128], zp[K // 8, N // 128]
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = offs_k[:, None] * stride_bzpk + (offs_n // group_size)[None, :] * stride_bzpn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    # Work on 8 rows once, this should be easily unrolled.
    for i in range(8):
        int4_b = ((int32_b << (28 - i * 4) >> 28) + 16) & 15
        int4_zp = ((zp_b << (28 - i * 4) >> 28) + 16) & 15
        bs_offs = (offs_k * 8 + i)[:, None] * stride_bsk + (offs_n // group_size)[None, :] * stride_bsn
        fpb_offs = (offs_k * 8 + i)[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
        k8_mask = (offs_k * 8 + i)[:, None] < K * 8
        scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask & k8_mask, other=0.0)
        fp_weight = (int4_b - int4_zp) * scale_b
        tl.store(fpb_ptr + fpb_offs, fp_weight, mask=n_mask & k8_mask)

def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    Kw, N = b.shape
    fp_b = torch.empty((b_scale.shape[0], b.shape[1]), device=device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(Kw, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        Kw, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b

def matmul_dequantize_int4(a, b, b_scale, b_zero_point, group_size=128, out=None):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=c)
    fp_b = None
    return c

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # You can fuse arbitrary activation functions here
    # accumulator = tl.math.max(accumulator, 0.0)
    
    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 1, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE_M=8,
        SPLIT_K=1,
        EVEN_K=True,
        ACC_TYPE=tl.float32,
    )
    return c

def quantize_int4(weight, group_size=128):
    """Quantize weight matrix to INT4 format."""
    assert weight.dim() == 2, "Weight must be a 2D tensor"
    rows, cols = weight.shape
    padded_rows = (rows + 7) // 8 * 8
    padded_cols = (cols + group_size - 1) // group_size * group_size
    
    # Pad weight if necessary
    if padded_rows > rows or padded_cols > cols:
        padded_weight = torch.zeros(padded_rows, padded_cols, device=weight.device, dtype=weight.dtype)
        padded_weight[:rows, :cols] = weight
    else:
        padded_weight = weight
    
    # Reshape for group-wise quantization
    grouped_weight = padded_weight.view(-1, group_size)
    
    # Compute scale and zero point
    scale = (grouped_weight.max(dim=-1)[0] - grouped_weight.min(dim=-1)[0]) / 15
    zero_point = grouped_weight.min(dim=-1)[0]
    
    # Quantize
    quantized = torch.clamp(torch.round((grouped_weight - zero_point.unsqueeze(-1)) / scale.unsqueeze(-1)), 0, 15).to(torch.int32)
    
    # Pack 8 INT4 values into one INT32
    packed = torch.zeros(padded_rows // 8, padded_cols, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= quantized[i::8, :] << (i * 4)
    
    return packed, scale.view(padded_rows, -1), zero_point.view(padded_rows, -1)

def unpack_int4(packed, scale, zero_point, original_shape):
    """Unpack INT4 matrix back to floating-point (for testing purposes)."""
    rows, cols = original_shape
    padded_rows = (rows + 7) // 8 * 8
    padded_cols = packed.shape[1]
    
    unpacked = torch.zeros(padded_rows, padded_cols, device=packed.device, dtype=torch.float32)
    for i in range(8):
        unpacked[i::8, :] = (packed & 0xF) * scale.unsqueeze(-1) + zero_point.unsqueeze(-1)
        packed >>= 4
    
    return unpacked[:rows, :cols]  # Remove padding
