triton
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    max_indices_ptr,
    n_elements,
    stride,
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)

    for i in range(start_idx, end_idx):
        x = tl.load(input_ptr + i * stride)
        sig_x = 1 / (1 + tl.exp(-x))
        tl.store(output_ptr + i * stride, sig_x)

        # Find the index of the maximum value
        max_val = tl.max(sig_x)
        max_index = tl.where(sig_x == max_val, i, -1)
        tl.store(max_indices_ptr + i * stride, max_index)

@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride,
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)

    for i in range(start_idx, end_idx):
        x = tl.load(input_ptr + i * stride)
        max_val = tl.max(x)
        max_index = tl.where(x == max_val, i, -1)
        tl.store(output_ptr + i * stride, max_index)

@triton.jit
def sigmoid_argmax_kernel(
    input_ptr,
    output_ptr,
    max_indices_ptr,
    n_elements,
    stride,
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)

    for i in range(start_idx, end_idx):
        x = tl.load(input_ptr + i * stride)
        sig_x = 1 / (1 + tl.exp(-x))
        tl.store(output_ptr + i * stride, sig_x)

        # Find the index of the maximum value
        max_val = tl.max(sig_x)
        max_index = tl.where(sig_x == max_val, i, -1)
        tl.store(max_indices_ptr + i * stride, max_index)

# Wrapper function
def sigmoid_argmax(input, dim=None, keepdim=False):
    import torch
    from triton.client import Client

    device = 'cuda' if input.device.type == 'cuda' else 'cpu'
    input_np = input.cpu().numpy()
    input_torch = torch.from_numpy(input_np).to(device)

    # Apply sigmoid
    sig_output = torch.sigmoid(input_torch)
    sig_output_np = sig_output.cpu().numpy()

    # Compute argmax
    if dim is None:
        argmax_result = torch.argmax(sig_output)
    else:
        argmax_result = torch.argmax(sig_output, dim=dim, keepdim=keepdim)

    return argmax_result.to(device)
