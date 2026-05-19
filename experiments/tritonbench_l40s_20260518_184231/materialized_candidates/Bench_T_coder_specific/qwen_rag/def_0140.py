import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr, B_ptr, C_ptr, n, m, k, alpha, beta, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    i = pid // m
    j = pid % m
    
    acc = tl.zeros((k,), dtype=tl.float32)
    
    for l in range(k):
        ai = i
        aj = l
        bi = l
        bj = j
        
        if ai < n and aj < n and bi < n and bj < m:
            a_value = tl.load(A_ptr + ai * n + aj)
            b_value = tl.load(B_ptr + bi * m + bj)
            
            if ai >= aj:  # Lower triangular part condition
                acc[l] += a_value * b_value
    
    acc *= alpha
    tl.store(C_ptr + i * m + j, acc * beta)
