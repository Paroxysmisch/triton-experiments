stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    activation: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_size_m)
    num_pid_n = tl.cdiv(n, block_size_n)
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m = min(num_pid_m - first_pid_m, group_size_m)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m)) % m
    offs_bn = (pid_n * block_size_n + tl.arange(0, block_size_n)) % n
    offs_k = tl.arange(0, block_size_k)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(k, block_size_k)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k * block_size_k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k * block_size_k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += block_size_k * stride_ak
        b_ptrs += block_size_k * stride_bk

    if activation == "relu":
        accumulator = relu(accumulator)

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0.0)

def matmul(a, b, activation=""):
    return jt.triton_call(matmul_kernel, a, b, a.shape[0], b.shape[1], a.shape[2], a.stride(0), a.stride(1), b.stride(0),
                         b.stride(1), a.dtype.element_ty, b.dtype.element_ty, activation, tiled=True, extern_kernels={
                             "relu": relu
                         })

class LinearLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, bias=None, activation=""):
        if A.shape[2] != B.shape[1]:
            raise ValueError(f"Incompatible dimensions in matrix multiplication. "
                             f"Expected A.shape[2] == B.shape[1] but got {A.shape[2]} != {B.shape[1]}")
        if bias is not None:
            if bias.shape[0] != A.shape[2]:
                raise ValueError(f"Shape mismatch between bias and input in batched matrix multiplication. "
                                 f"Expected bias.shape[0] == A.shape[2] but got {bias.shape[0]} != {A.shape[2]}")
            if bias.ndim != 1:
                raise ValueError(f"bias must be a 1D tensor but got bias.ndim == {bias.ndim}")
            bias = bias.unsqueeze(1)
        out = _matmul_fused_bias_act(A, B, bias, activation)
        ctx.save_for_backward(A, B, bias)
        ctx.activation = activation
        return out

def linear_layer(A, B, bias=None, activation=""):
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)
    return LinearLayer.apply(A, B, bias, activation)
