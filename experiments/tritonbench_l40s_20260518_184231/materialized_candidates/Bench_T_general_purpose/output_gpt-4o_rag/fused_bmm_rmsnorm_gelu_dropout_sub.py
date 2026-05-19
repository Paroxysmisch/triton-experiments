import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def rms_norm_kernel(output_ptr, input_ptr, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    input_ptr += row_idx * stride
    output_ptr += row_idx * stride

    tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tmp += a * a

    rms = tl.sqrt(tl.sum(tmp) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        tl.store(output_ptr + cols, x_hat, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Step 1: Batch Matrix Multiplication
    Z = torch.bmm(input1, input2)  # (B, N, P)

    # Step 2: RMS Normalization using Triton
    B, N, P = Z.shape
    Z_norm = torch.empty_like(Z)
    BLOCK_SIZE = triton.next_power_of_2(P)
    grid = (B * N,)
    rms_norm_kernel[grid](Z_norm, Z, Z.stride(0), P, eps, BLOCK_SIZE=BLOCK_SIZE)

    # Step 3: GELU Activation
    if approximate == 'tanh':
        G = F.gelu(Z_norm, approximate='tanh')
    else:
        G = F.gelu(Z_norm)

    # Step 4: Dropout
    if training:
        D = F.dropout(G, p=dropout_p, training=True)
    else:
        D = G

    # Step 5: Subtraction
    Y = D - other

    if out is not None:
        out.copy_(Y)
        return out

    return Y

# Example usage:
B, N, M, P = 32, 64, 128, 256
input1 = torch.randn(B, N, M, device='cuda')
input2 = torch.randn(B, M, P, device='cuda')
other = torch.randn(B, N, P, device='cuda')

output = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape=P)
print(output.shape)  # Should print: torch.Size([32, 64, 256])
