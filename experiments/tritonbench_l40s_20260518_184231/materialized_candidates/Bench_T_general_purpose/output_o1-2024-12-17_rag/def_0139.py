import triton
import triton.language as tl
import torch

@triton.jit
def _std_reduce_kernel(
    input_ptr,  # pointer to input data
    sum_ptr,    # pointer to partial sum
    sumsq_ptr,  # pointer to partial sum of squares
    N,          # number of elements to reduce
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x * x, axis=0)

    if tl.thread_id_x() == 0:
        tl.store(sum_ptr + pid, sum_val)
        tl.store(sumsq_ptr + pid, sum_sq_val)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    """
    Calculates the standard deviation of the input tensor along the specified dimension(s).
    Args:
        input (Tensor): input tensor
        dim (int or tuple of ints, optional): dimension(s) to reduce
        correction (int, optional): difference between the sample size and degrees of freedom (Bessel's correction)
        keepdim (bool, optional): whether the output tensor has dim retained or not
        out (Tensor, optional): the output tensor
    Returns:
        Tensor: standard deviation of input along specified dimension(s)
    """
    # Handle dim=None => reduce over all dimensions
    if dim is None:
        # Flatten input
        reduced_input = input.view(-1)
        N = reduced_input.numel()
        # Launch kernel in chunks
        BLOCK_SIZE = 1024
        num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

        sum_buf = torch.empty(num_blocks, dtype=torch.float32, device=reduced_input.device)
        sum_sq_buf = torch.empty(num_blocks, dtype=torch.float32, device=reduced_input.device)

        _std_reduce_kernel[(num_blocks,)](
            reduced_input, sum_buf, sum_sq_buf, N, BLOCK_SIZE=BLOCK_SIZE
        )

        # Finalize sums on CPU
        total_sum = sum_buf.sum().item()
        total_sum_sq = sum_sq_buf.sum().item()

        mean_val = total_sum / N
        var_unbiased = (total_sum_sq / N) - (mean_val * mean_val)

        # Apply correction if possible
        if N > correction:
            var_unbiased *= (N / (N - correction))

        std_val = var_unbiased**0.5
        result = torch.tensor(std_val, device=reduced_input.device, dtype=input.dtype)
        if keepdim:
            # Expand back to original shape of all ones
            result = result.view([1]*input.dim())

    else:
        # Normalize dim into a tuple
        if isinstance(dim, int):
            dim = (dim,)
        # Sort dims to reduce from highest to lowest
        dim = sorted(list(dim))
        # Temporarily transpose / reshape to reduce one dim at a time
        result = input
        for d in reversed(dim):
            size_d = result.size(d)
            # Merge the dimension d into a single axis for kernel reduction
            other_dims = list(result.shape)
            merged_size = 1
            for idx, sz in enumerate(other_dims):
                if idx == d:
                    merged_size = sz
                    break

            # Reshape so that we reduce over the last axis
            perm_order = list(range(result.dim()))
            perm_order[d], perm_order[-1] = perm_order[-1], perm_order[d]
            result = result.permute(perm_order)
            new_shape = result.shape[:-1]
            N = result.shape[-1]
            flattened = result.reshape(-1, N)

            BLOCK_SIZE = 1024
            num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

            sums = []
            sums_sq = []
            # Reduce each row of flattened
            out_rows = []
            for row_idx in range(flattened.size(0)):
                row_data = flattened[row_idx]
                sum_buf = torch.empty(num_blocks, dtype=torch.float32, device=row_data.device)
                sum_sq_buf = torch.empty(num_blocks, dtype=torch.float32, device=row_data.device)

                _std_reduce_kernel[(num_blocks,)](
                    row_data, sum_buf, sum_sq_buf, N, BLOCK_SIZE=BLOCK_SIZE
                )
                row_sum = sum_buf.sum().item()
                row_sum_sq = sum_sq_buf.sum().item()

                mean_val = row_sum / N
                var_unbiased = (row_sum_sq / N) - (mean_val * mean_val)
                if N > correction:
                    var_unbiased *= (N / (N - correction))
                std_val = var_unbiased**0.5
                out_rows.append(std_val)

            # Reshape to new_shape after reduction for this dim
            tensor_out = torch.tensor(out_rows, device=result.device, dtype=result.dtype)
            tensor_out = tensor_out.view(new_shape)

            # Permute dimensions back
            perm_back = list(range(len(perm_order)))
            perm_back[-1], perm_back[d] = perm_back[d], perm_back[-1]
            tensor_out = tensor_out.permute(perm_back)

            # Conditionally keep dim
            if keepdim:
                # re-insert dimension of size 1
                shape_list = list(tensor_out.shape)
                shape_list[d] = 1
                tensor_out = tensor_out.view(shape_list)

            result = tensor_out

        # final result
    if out is not None:
        out.copy_(result)
        return out
    return result
