import torch
import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def silu(x):
        # Using tl.sigmoid to approximate silu, may lead to lower performance
        # return x * tl.sigmoid(x)
        # Use fast approximation
        return x * 0.5 * (1.0 + tl.math.erf(x / 1.41421356237))

    def calculate_settings(
        BLOCK_SIZE: int, MAX_FUSED_SIZE: int, MULITPROCESS: bool = False
    ):
        if MULITPROCESS:
            if torch.cuda.get_device_properties(0).name == "HIP":
                num_warps = 4
            else:
                num_warps = 8
        else:
            num_warps = 4
        return BLOCK_SIZE, num_warps

    @triton.jit
    def _swiglu_forward_kernel(
        a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE: tl.constexpr, silu: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        a_row_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, N),
            strides=(1, N),
            offsets=(pid * BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, N),
            order=(0, 1),
        )
        a_block = tl.load(a_row_block_ptr)
        a_silu = silu(a_block)
        b_block = tl.load(b_ptr + pid * BLOCK_SIZE : pid * BLOCK_SIZE + b_ptr + N)
        c_block = a_silu * b_block
        c_ptr_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(1, N),
            offsets=(pid * BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, N),
            order=(0, 1),
        )
        tl.store(c_ptr_block_ptr, c_block)

    def swiglu_forward(a, b):
        M, N = a.shape
        c = torch.empty_like(a)
        MAX_FUSED_SIZE = 65536 // a.element_size()
        BLOCK_SIZE, num_warps = calculate_settings(
            min(MAX_FUSED_SIZE, triton.next_power_of_2(M)), MAX_FUSED_SIZE
        )
        _swiglu_forward_kernel[(M + BLOCK_SIZE - 1) // BLOCK_SIZE,](
            a, b, c, M, N, num_warps=num_warps, silu=silu
        )
        return c

    @triton.jit
    def _swiglu_backward_kernel(
        a_ptr, b_ptr, da_ptr, db_ptr, M, N, BLOCK_SIZE: tl.constexpr, silu: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        a_row_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, N),
            strides=(1, N),
            offsets=(pid * BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, N),
            order=(0, 1),
        )
        a_block = tl.load(a_row_block_ptr)
        a_silu_grad = silu(a_block)
        b_block = tl.load(b_ptr + pid * BLOCK_SIZE : pid * BLOCK_SIZE + b_ptr + N)
        da_block = a_silu_grad * b_block
        db_block = a_block * b_block
        da_ptr_block_ptr = tl.make_block_ptr(
            base=da_ptr,
            shape=(M, N),
            strides=(1, N),
            offsets=(pid * BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, N),
            order=(0, 1),
        )
        tl.store(da_ptr_block_ptr, da_block)
        db_ptr_block_ptr = tl.make_block_ptr(
            base=db_ptr,
            shape=(M, N),
            strides=(1, N),
            offsets=(pid * BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, N),
            order=(0, 1),
        )
        tl.store(db_ptr_block_ptr, db_block)

    def swiglu_backward(a, b, da, db):
        M, N = a.shape
        MAX_FUSED_SIZE = 65536 // a.element_size()
        BLOCK_SIZE, num_warps = calculate_settings(
            min(MAX_FUSED_SIZE, triton.next_power_of_2(M)), MAX_FUSED_SIZE
        )
        _swiglu_backward_kernel[(M + BLOCK_SIZE - 1) // BLOCK_SIZE,](
            a, b, da, db, M, N, num_warps=num_warps, silu=silu
        )
        return da, db
