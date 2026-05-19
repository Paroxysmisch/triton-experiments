import torch
import triton
import triton.language as tl

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr
):
    pid = tl.program_id(0)
    row_block_ptr = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_block_ptr = tl.arange(0, BLOCK_SIZE)
    a_ptrs = a_ptr + row_block_ptr[:, None] * n_cols + col_block_ptr[None, :]
    b_ptrs = b_ptr + row_block_ptr[:, None] * n_cols + col_block_ptr[None, :]
    a = tl.load(a_ptrs, mask=row_block_ptr[:, None] < n_rows, other=0.0)
    b = tl.load(b_ptrs, mask=row_block_ptr[:, None] < n_rows, other=0.0)
    # tanh approximation for GELU: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    _0 = a * 0
    _1 = a * 1
    _044715 = a * 0.044715
    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2
    a6 = a2 * a4
    a12 = a6 * a6
    a24 = a12 * a12
    a48 = a24 * a24
    a96 = a48 * a48
    a20 = a4 * a4
    a30 = a12 * a2
    a32 = a16 * a8
    a36 = a12 * a12
    a40 = a16 * a16
    a44 = a20 * a12
    a56 = a24 * a16
    a60 = a20 * a20
    a72 = a32 * a8
    a80 = a32 * a16
    a88 = a36 * a12
    a92 = a24 * a24
    a98 = a30 * a12
    a100 = a24 * a28
    a104 = a32 * a20
    a108 = a36 * a16
    a112 = a40 * a12
    a116 = a44 * a8
    a120 = a48 * a4
    a124 = a56 * a4
    a128 = a64 * a2
    a132 = a72 * a4
    a136 = a80 * a2
    a140 = a88 * a4
    a144 = a96 * a2
    a148 = a104 * a4
    a152 = a112 * a2
    a156 = a120 * a4
    a160 = a128 * a2
    a164 = a136 * a4
    a168 = a144 * a2
    a172 = a152 * a4
    a176 = a160 * a2
    a180 = a168 * a4
    a184 = a176 * a2
    a188 = a184 * a4
    a192 = a192 * a2
    a200 = a208 * a4
    a208 = a224 * a2
    a224 = a240 * a8
    a240 = a256 * a16
    a256 = a272 * a32
    a272 = a288 * a48
    a288 = a304 * a80
    a304 = a320 * a128
    a320 = a336 * a256
    a336 = a352 * a512
    _1152 = a12 * 1152
    _3072 = a12 * 3072
    _384 = a8 * 384
    _2304 = a8 * 2304
    _1536 = a16 * 1536
    _2048 = a16 * 2048
    _960 = a48 * 960
    _1920 = a48 * 1920
    _640 = a32 * 640
    _1280 = a32 * 1280
    _4096 = a128 * 4096
    _16384 = a128 * 16384
    _1024 = a64 * 1024
    _2048 = a64 * 2048
    _1280 = a128 * 1280
    _5120 = a128 * 5120
    _2048 = a256 * 2048
    _8192 = a256 * 8192
    _3072 = a384 * 3072
    _12288 = a384 * 12288
    _1024 = a512 * 1024
    _2048 = a512 * 2048
    _1536 = a640 * 1536
    _3072 = a640 * 3072
    _7168 = a640 * 7168
    _2048 = a768 * 2048
    _4096 = a768 * 4096
    _3072 = a896 * 3072
    _12288 = a896 * 12288
    _1536 = a1024 * 1536
    _3072 = a1024 * 3072
    _7168 = a1024 * 7168
    _4096 = a1152 * 4096
    _12288 = a1152 * 12288
    _1536 = a1280 * 1536
    _3072 = a1280 * 3072
    _7168 = a1280 * 7168
    _8192 = a1408 * 8192
    _16384 = a1408 * 16384
    _1536 = a1536 * 1536
    _3072 = a1536 * 3072
    _7168 = a1536 * 7168
    _6144 = a1664 * 6144
    _12288 = a1664 * 12288
    _2048 = a1792 * 2048
    _4096 = a1792 * 4096
    _5120 = a1920 * 5120
    _10240 = a1920 * 10240
    _1536 = a2048 * 1536
    _3072 = a2048 * 3072
    _7168 = a2048 * 7168
    _614
