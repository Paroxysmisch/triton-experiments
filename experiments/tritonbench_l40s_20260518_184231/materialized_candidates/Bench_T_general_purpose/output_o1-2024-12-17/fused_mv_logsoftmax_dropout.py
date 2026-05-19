import triton
import triton.language as tl
import torch

# Kernel 1: Compute matrix-vector multiplication and partial sum of exp(z)
@triton.jit
def _mv_logsoftmax_sumexp_kernel(
    A_ptr,       # [n, m] float32
    V_ptr,       # [m] float32
    Z_ptr,       # [n] float32 (to store dot product results)
    EZ_ptr,      # [n] float32 (to store exp(dot product) for each row)
    SUMEXP_ptr,  # [1] float32 (to accumulate sum of exp of all rows)
    n, m,
    BLOCK_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)  # Each program handles exactly one row (pid)
    # Check bounds
    if pid >= n:
        return

    # Dot product over row pid
    row_start = pid * m
    # We'll use a loop over columns with a step of BLOCK_SIZE_M
    # Accumulate the partial sum
    dot = tl.float32(0)
    offs = tl.arange(0, BLOCK_SIZE_M)
    for col_i in range(0, m, BLOCK_SIZE_M):
        cols = col_i + offs
        mask = cols < m
        A_val = tl.load(A_ptr + row_start + cols, mask=mask, other=0.0)
        V_val = tl.load(V_ptr + cols, mask=mask, other=0.0)
        dot += tl.sum(A_val * V_val, where=mask)

    # Store z[pid] = dot
    tl.store(Z_ptr + pid, dot)
    # Compute exp(dot)
    e_dot = tl.exp(dot)
    # Store e^z in EZ_ptr
    tl.store(EZ_ptr + pid, e_dot)
    # Atomic add to the global sumExp
    tl.atomic_add(SUMEXP_ptr, e_dot)


# Kernel 2: Final log-softmax and optional dropout
@triton.jit
def _logsoftmax_dropout_kernel(
    Z_ptr,       # [n] float32 (dot product results)
    SUMEXP_ptr,  # [1] float32 (sum of exp(z))
    OUT
