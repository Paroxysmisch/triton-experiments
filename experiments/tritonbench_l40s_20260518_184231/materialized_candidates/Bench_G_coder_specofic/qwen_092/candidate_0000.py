import triton
import triton.language as tl

# Constants
BLOCK = 64
NUM_BLOCK = 1  # Assuming NUM_BLOCK is 1 for simplicity
CBLOCK = 32
NUM_CBLOCK = BLOCK // CBLOCK

@triton.jit
def _fwd_kernel(
    Q, K, V, DO, SO,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    b, h, i, j = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    q = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    k = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK)
    v = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    do = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    so = tl.zeros_like(do)

    for k in range(NUM_BLOCK):
        for c in range(NUM_CBLOCK):
            q_block = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            k_block = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            v_block = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            do_block = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            so_block = tl.zeros_like(do_block)

            for l in range(CBLOCK):
                for m in range(CBLOCK):
                    dot = tl.dot(q_block[l], k_block[m])
                    so_block[l] += dot * v_block[m]

            tl.store(SO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, so_block)

    tl.store(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK, so)

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, DO, DQ, DK, DV,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    b, h, i, j = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    q = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    k = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK)
    v = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    do = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    dq = tl.zeros_like(q)
    dk = tl.zeros_like(k)
    dv = tl.zeros_like(v)

    for k in range(NUM_BLOCK):
        for c in range(NUM_CBLOCK):
            q_block = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            k_block = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            v_block = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            do_block = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            dq_block = tl.zeros_like(q_block)
            dk_block = tl.zeros_like(k_block)
            dv_block = tl.zeros_like(v_block)

            for l in range(CBLOCK):
                for m in range(CBLOCK):
                    dot = tl.dot(q_block[l], k_block[m])
                    dq_block[l] += do_block[m] * k_block[m]
                    dk_block[m] += do_block[l] * q_block[l]
                    dv_block[m] += do_block[l] * k_block[m]

            tl.store(DQ + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dq_block)
            tl.store(DK + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dk_block)
            tl.store(DV + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dv_block)

    tl.store(DQ + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK, dq)
    tl.store(DK + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK, dk)
    tl.store(DV + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + j * BLOCK, dv)

@triton.jit
def _bwd_inter_kernel(
    Q, K, V, DO, DQ, DK, DV,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    b, h, i, j = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    q = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    k = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK)
    v = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    do = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK)
    dq = tl.zeros_like(q)
    dk = tl.zeros_like(k)
    dv = tl.zeros_like(v)

    for k in range(NUM_BLOCK):
        for c in range(NUM_CBLOCK):
            q_block = tl.load(Q + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            k_block = tl.load(K + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            v_block = tl.load(V + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            do_block = tl.load(DO + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK)
            dq_block = tl.zeros_like(q_block)
            dk_block = tl.zeros_like(k_block)
            dv_block = tl.zeros_like(v_block)

            for l in range(CBLOCK):
                for m in range(CBLOCK):
                    dot = tl.dot(q_block[l], k_block[m])
                    dq_block[l] += do_block[m] * k_block[m]
                    dk_block[m] += do_block[l] * q_block[l]
                    dv_block[m] += do_block[l] * k_block[m]

            tl.store(DQ + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dq_block)
            tl.store(DK + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dk_block)
            tl.store(DV + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + k * CBLOCK * CBLOCK + c * CBLOCK, dv_block)

    tl.store(DQ + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK, dq)
    tl.store(DK + b * h * BLOCK * BLOCK + j * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK, dk)
    tl.store(DV + b * h * BLOCK * BLOCK + i * BLOCK * BLOCK + j * BLOCK * BLOCK + j * BLOCK, dv)

@triton.jit
def _fwd_wrapper(
    Q, K, V, DO, SO
