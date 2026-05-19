import torch
import triton
import triton.language as tl
from ..utils.shape_utils import can_use_int32_index


@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    # Load input data
    inp_ptrs = inp + offset
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))

    # Compute max and index
    max_val = tl.max(inp_val)
    max_index = tl.argmax(inp_val, axis=0)

    # Store intermediate results
    mid_index_ptrs = mid_index + pid
    mid_value_ptrs = mid_value + mid_index_ptrs
    tl.store(mid_value_ptrs, max_val, mask=(not INT64_INDEX))
    tl.store(mid_index_ptrs, max_index, mask=INT64_INDEX)


@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    # Load intermediate data
    mid_value_ptrs = mid_value + offset
    mid_index_ptrs = mid_index + offset
    mid_val = tl.load(mid_value_ptrs, mask=mask, other=-float("inf"))
    mid_index_val = tl.load(mid_index_ptrs, mask=mask)

    # Compute final max and index
    max_val = tl.max(mid_val)
    max_index = tl.argmax(mid_val, axis=0)

    # Resolve index conflict
    index_conflict = tl.where(
        mid_index_val == max_index, 0, mid_index_val * 0 - 1
    )
    max_global_index = tl.sum(index_conflict)

    # Store output
    out_ptrs = out + offset
    tl.store(out_ptrs, max_global_index, mask=(not INT64_INDEX))
    tl.store(out_ptrs + 1, max_val, mask=(INT64_INDEX and mask[0]))


def argmax(inp, dim=None, keepdim=False):
    inp_shape = tuple(inp.shape)
    inp_dim = len(inp_shape)

    if dim is None:
        # Flatten input and compute argmax
        inp = inp.contiguous()
        M = inp.numel()
        INT64_INDEX = not can_use_int32_index(M)
        out = torch.empty(2, 1, dtype=torch.int64, device=inp.device) if INT64_INDEX else torch.empty(1, 1, dtype=torch.int64, device=inp.device)
        BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        BLOCK_MID = triton.next_power_of_2(math.ceil(math.sqrt(M // 2)))

        mid_value = torch.empty((1, 1), dtype=torch.int64, device=inp.device)
        mid_index = torch.empty(
            mid_value.shape, dtype=torch.int64 if INT64_INDEX else torch.int32, device=inp.device
        )

        # Compute mid results
        grid_mid = (triton.cdiv(M, BLOCK_SIZE * 2), 1)
        argmax_kernel_1[grid_mid](
            inp, mid_value, mid_index, M, BLOCK_SIZE, INT64_INDEX
        )

        # Compute final result
        grid_final = (1, 1)
        argmax_kernel_2[grid_final](mid_value, mid_index, out, mid_value.numel(), BLOCK_MID)

        out = out.squeeze()
        if not keepdim:
            out = out.unsqueeze(0)
        return out

    else:
        # Compute argmax across a specified dimension
        assert dim >= -inp_dim and dim < inp_dim, "Invalid dim"
        dim = dim % inp_dim
        M = 1
        N = inp_shape[dim]
        K = inp.numel() // N

        inp = inp.contiguous()
        out = torch.empty(
            2, K if keepdim else 1, dtype=torch.int64, device=inp.device
        )

        BLOCK_M = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        BLOCK_N = triton.next_power_of_2(math.ceil(math.sqrt(N)))

        if not keepdim:
            out = out.squeeze(1)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_N"]),
        )
        argmax_kernel[grid](
            inp, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
        )
        return out.squeeze(0)
