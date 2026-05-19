import numpy as np
import triton
import triton.language as tl

# Define Triton types
INT64 = triton.types.int64
INT32 = triton.types.int32
FLOAT32 = triton.types.float32

@triton.jit
def argmax_kernel_1(inp, mid_value, mid_index, M, BLOCK_SIZE, INT64_INDEX):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE
    mask = offsets < M
    if INT64_INDEX:
        inp = inp.bitcast(INT64)
    inp = inp + offsets
    v = inp[0]
    i = 0
    for offset in tl.arange(1, BLOCK_SIZE):
        update_mask = (inp[offset] > v) & mask
        v = tl.where(update_mask, inp[offset], v)
        i = tl.where(update_mask, offset, i)
    mid_value[pid] = v
    mid_index[pid] = i

@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID):
    pid = tl.program_id(axis=0)
    if pid < mid_size:
        v = mid_value[pid]
        i = mid_index[pid]
        for offset in tl.arange(BLOCK_MID, mid_size):
            update_mask = mid_value[offset] > v
            v = tl.where(update_mask, mid_value[offset], v)
            i = tl.where(update_mask, mid_index[offset], i)
        out[pid] = i

def can_use_int32_index(M):
    return M <= np.iinfo(np.int32).max

def argmax(x, dim=None):
    x = triton.testing.numpy_to_torch(x)
    if dim is None:
        x = x.reshape(-1)
        M = x.numel()
        dtype = x.dtype
        x = x.cuda()
        mid_value = torch.empty((M - 1) // 256 + 1, dtype=dtype, device=x.device)
        mid_index = torch.empty_like(mid_value)
        grid = lambda M: ((M + 255) // 256, )
        argmax_kernel_1[grid](x, mid_value, mid_index, M, 256, can_use_int32_index(M))
        argmax_kernel_2[grid](mid_value, mid_index, x, mid_value.numel(), 256)
        return x[0].item()
    else:
        assert dim in (0, 1)
        x = x.transpose(0, dim)
        M, N = x.shape
        x = x.reshape(M * N)
        dtype = x.dtype
        x = x.cuda()
        mid_value = torch.empty((M * (N - 1) + 1) // 256, dtype=dtype, device=x.device)
        mid_index = torch.empty_like(mid_value)
        grid = lambda M: ((M + 255) // 256, )
        argmax_kernel_1[grid](x, mid_value, mid_index, M * N, 256, can_use_int32_index(M * N))
        argmax_kernel_2[grid](mid_value, mid_index, x, mid_value.numel(), 256)
        index = x.argmax().item()
        return index // N, index % N
