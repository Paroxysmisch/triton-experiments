import triton
import triton.language as tl

# Helper functions
@triton.jit
def bmm(input1, input2, out):
    # Batch Matrix Multiplication
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    k = tl.program_id(axis=2)
    
    acc = tl.zeros((8,), dtype=tl.float32)
    for kk in range(tl.cdiv(input1.shape[2], 8)):
        acc += input1[i, j, kk * 8:(kk + 1) * 8] * input2[kk * 8:(kk + 1) * 8, k]
    out[i, j, k] = acc

@triton.jit
def rms_norm(x, gamma, eps):
    # RMS Normalization
    m = tl.mean(x * x, axis=-1, keepdims=True)
    r = tl.sqrt(m + eps)
    return x / r * gamma

@triton.jit
def gelu(x, approximate):
    # GELU Activation
    if approximate == 'none':
        return x * 0.5 * (1.0 + tl.tanh((x * 0.79788456) + (0.044715 * (x * x * x * x))))
    elif approximate == 'tanh':
        return 0.5 * x * (1.0 + tl.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * (x * x * x * x))))))
    else:
        raise ValueError("Invalid approximate method")

@triton.jit
def dropout(x, p, training):
    # Dropout
    if training:
        mask = tl.random.rand_like(x) > p
        return x * mask / (1.0 - p)
    else:
        return x

# Main fused operation
@triton.jit
def fused_bmm_rmsnorm_gelu_dropout(input1, input2, gamma, eps, p, training, approximate, out):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    k = tl.program_id(axis=2)
    
    z1 = tl.zeros((8,), dtype=tl.float32)
    for kk in range(tl.cdiv(input1.shape[2], 8)):
        z1 += input1[i, j, kk * 8:(kk + 1) * 8] * input2[kk * 8:(kk + 1) * 8, k]
    
    z2 = rms_norm(z1, gamma, eps)
    z3 = gelu(z2, approximate)
    z = dropout(z3, p, training)
    
    out[i, j, k] = z

# Wrapper function
def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', out=None):
    B, N, M = input1.shape
    _, M, P = input2.shape
    
    if out is None:
        out = tl.zeros((B, N, P), dtype=input1.dtype)
    
    grid = (B, N, P)
    block = (8, 1, 1)
    
    fused_bmm_rmsnorm_gelu_dropout[grid, block](input1, input2, gamma=1.0, eps=eps, p=dropout_p, training=training, approximate=approximate, out=out)
    
    return out
