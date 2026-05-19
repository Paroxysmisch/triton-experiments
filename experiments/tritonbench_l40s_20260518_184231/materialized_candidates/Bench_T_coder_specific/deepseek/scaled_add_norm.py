@triton.jit
def scaled_add_norm(y_ptr, x_ptr, alpha, n):
    y = t2h[y_ptr]
    x = t2h[x_ptr]
    y += alpha * x
    norm = torch.norm(y, 2)
    h2t[y_ptr] = y
    return norm
