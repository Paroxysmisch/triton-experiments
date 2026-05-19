import triton.language as tl

@triton.jit
def zeta(x_ptr, q_ptr, output_ptr, N):
    x_mask = tl.program_id() < N
    x = tl.load(x_ptr, mask=x_mask, other=0)
    q = tl.load(q_ptr, mask=x_mask, other=0)
    sum = 0.0
    k = tl.program_id()
    while k < N:
        sum += 1 / (k + q) ** x
        k += tl.program_id()
    output = tl.load(output_ptr, mask=x_mask, other=0) + sum
    tl.store(output_ptr, output, mask=x_mask)
