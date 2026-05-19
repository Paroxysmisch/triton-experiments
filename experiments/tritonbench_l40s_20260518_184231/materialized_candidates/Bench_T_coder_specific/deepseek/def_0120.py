@triton.jit
def matrix_vector_dot_kernel(A_ptr, A_stride, x_ptr, y_ptr, y_stride, alpha, beta, n, m, DTYPE):
    pid = tl.program_id(axis=0)
    mask = pid < n
    x = tl.load(x_ptr + pid * DTYPE.itemsize)
    y = tl.load(y_ptr + pid * DTYPE.itemsize)
    dot = alpha * x * y + beta * y
    for i in range(m):
        A_val = tl.load(A_ptr + pid * A_stride + i * DTYPE.itemsize)
        dot += A_val * x
    tl.store(y_ptr + pid * DTYPE.itemsize, dot)

def matrix_vector_dot(A, x, y, alpha, beta):
    n, m = A.shape
    y = y.to(device='cuda')
    A = A.to(device='cuda')
    x = x.to(device='cuda')
    DTYPE = torch.float32
    y_ptr = y.data_ptr()
    A_ptr = A.data_ptr()
    x_ptr = x.data_ptr()
    A_stride = A.stride(1)
    y_stride = y.stride(0)
    grid = lambda : triton.cuda.grid(1, group_size=n)
    matrix_vector_dot_kernel[grid](A_ptr, A_stride, x_ptr, y_ptr, y_stride, alpha, beta, n, m, DTYPE)
    y = y.to(device='cpu')
    return torch.dot(y, x)
