import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    square_output = row * row
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)

@triton.jit
def mean_of_squares_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()
    square_output = row * row
    mean_output = tl.sum(square_output) / n_cols + eps
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, mean_output, mask=col_offsets < n_cols)

@triton.jit
def rms_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()
    square_output = row * row
    rms = tl.sqrt(tl.sum(square_output) / n_cols + eps)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, rms, mask=col_offsets < n_cols)

@triton.jit
def rms_norm(output_ptr, input_ptr, weights_ptr, stride, N, eps, DTYPE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    output_ptr += row * stride
    input_ptr += row * stride
    tmp = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        tmp += a * a
    rms = tl.sqrt(tl.sum(tmp) / N + eps)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(DTYPE)
        w = tl.load(weights_ptr + cols, mask=mask)
        x_hat = x / rms
        y = x_hat * w
        tl.store(output_ptr + cols, y, mask=mask)

class RMSNormTriton(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor):
        return self._rms_norm(x, self.weight, self.eps).type_as(x)
    def _rms_norm(self, x, w, eps):
        x_reshaped = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_reshaped.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        data_type = x.dtype
        if data_type == torch.float32:
            data_type = tl.float32
        elif data_type == torch.float16:
            data_type = tl.float16
        elif data_type == torch.int64:
            data_type = tl.int64
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        y = torch.empty_like(x_reshaped)
        rms_norm[(n_rows,)](
            y,
            x_reshaped,
            w,
            x_reshaped.stride(0),
            n_cols,
            eps,
            DTYPE=data_type,
            BLOCK_SIZE=BLOCK_SIZE
        )
        y = y.reshape(*x.shape)
        return y

def fused_bmm_rmsnorm_gelu_dropout_sub(input1,
                                       input2,
                                       other,
                                       normalized_shape,
                                       dropout_p=0.5,
                                       training=True,
                                       approximate='none',
                                       eps=1e-5,
                                       *,
                                       out=None):
    Z = torch.bmm(input1, input2)  # (B, N, P)
    if isinstance(normalized_shape, int):
        dim = normalized_shape
    else:
        dim = normalized_shape[-1] if isinstance(normalized_shape, (list, tuple)) else normalized_shape

    rmsnorm_module = RMSNormTriton(dim, eps)
    Z_norm = rmsnorm_module(Z)
    G = F.gelu(Z_norm, approximate=approximate)
    D = F.dropout(G, p=dropout_p, training=training)
    Y = D - other
    if out is not None:
        out.copy_(Y)
        return out
    return Y
