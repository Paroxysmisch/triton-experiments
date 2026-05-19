# Triton wrapper
@triton.jit
def batch_mm(A, B, C, n, m, p):
    # Your Triton kernel code here
    pass

# Wrapper function
def batch_mm(A, B):
    assert A.dim() == 3 and B.dim() == 3
    assert A.size(0) == B.size(0)
    assert A.size(2) == B.size(1)
    C = torch.empty((A.size(0), A.size(1), B.size(2)), device=A.device)
    batch_mm[A.shape[0], A.shape[1], A.shape[2]](A, B, C)
    return C
