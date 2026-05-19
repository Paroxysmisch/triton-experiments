import triton
import triton.language as tl
import torch

@triton.jit
def adaptive_avg_pool2d_kernel(input_ptr, output_ptr, H_in, W_in, S_0, S_1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n = pid // S_1
    w_out = pid % S_1

    for h_out in range(S_0):
        h_start = h_out * H_in // S_0
        h_end = (h_out + 1) * H_in // S_0
        w_start = w_out * W_in // S_1
        w_end = (w_out + 1) * W_in // S_1

        sum = 0.0
        count = 0

        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                idx = n * H_in * W_in + h * W_in + w
                sum += tl.load(input_ptr + idx)
                count += 1

        avg = sum / count
        out_idx = n * S_0 * S_1 + h_out * S_1 + w_out
        tl.store(output_ptr + out_idx, avg)

def adaptive_avg_pool2d(input, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    H_in, W_in = input.shape[-2], input.shape[-1]
    S_0, S_1 = output_size

    if S_0 is None:
        S_0 = H_in
    if S_1 is None:
        S_1 = W_in

    N, C = input.shape[0], input.shape[1] if len(input.shape) == 4 else (1, input.shape[0])
    output = torch.empty((N, C, S_0, S_1), device=input.device, dtype=input.dtype)

    grid = (N * S_1,)
    triton_input = input.flatten().contiguous()
    triton_output = output.flatten().contiguous()

    adaptive_avg_pool2d_kernel[grid](
        triton_input,
        triton_output,
        H_in,
        W_in,
        S_0,
        S_1,
        BLOCK_SIZE=1024
    )

    return output
