@triton.jit
def _polygamma(OUT, input, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_N)
    x = input[pid * BLOCK_N + idx]
    # Here, you need to implement the digamma function and its n-th derivative using Triton language
    result = tl.polygamma(n, x)
    OUT[pid * BLOCK_N + idx] = result
