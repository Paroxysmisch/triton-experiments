import torch
import triton
import triton.language as tl

# Simple Philox-based RNG utility
@triton.jit
def philox_engine(key, idx):
    # Constants for Philox 4x32-10
    ROUNDS = 10
    MUL_LO = 0xD2511F53
    MUL_HI = 0xCD9E8D57

    x = tl.uint32(idx)
    y = tl.uint32(idx >> 32)
    z = tl.uint32(key)
    w = tl.uint32(key >> 32)

    # 10-round Philox
    for _ in range(ROUNDS):
        # Multiply high and low
        hi_x = tl.mul_hi(x, MUL_LO)
        lo_x = tl.mul_lo(x, MUL_LO)
        hi_y = tl.mul_hi(y, MUL_HI)
        lo_y = tl.mul_lo(y, MUL_HI)

        x = hi_y ^ z
        y = hi_x ^ w
        z += tl.uint32(0x9E3779B9)
        w += tl.uint32(0xBB67AE85)

        # Swap lo_x, lo_y to keep generating
        x, y = y, x
        lo_x, lo_y = lo_y, lo_x

    return tl.make_tuple(lo_x, lo_y, x, y)

@triton.jit
def fill_rand_kernel(
    ptr_out, # output pointer
    seed,    # philox seed
    offset,  # global index offset
    n_elems, # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elems

    # Generate random 32-bit data, convert to float32 [0,1)
    # Combine seed and offset to form key and idx for Philox
    key = tl.uint64(seed)
    idx = tl.uint64(offsets + offset)
    lo_x, _, _, _ = philox_engine(key, idx)

    rnd_val = (lo_x.to(tl.float32) * (1.0 / 4294967296.0))
    tl.store(ptr_out + offsets, rnd_val, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided,
         device=None, requires_grad=False, pin_memory=False):
    shape = size[0] if isinstance(size[0], (list, tuple)) else size
    if out is None:
        out = torch.empty(shape, dtype=dtype if dtype else torch.get_default_dtype(),
                          layout=layout, device=device if device else torch.device('cpu'),
                          requires_grad=requires_grad, pin_memory=pin_memory)
    # Flatten
    flat_out = out.view(-1)
    n_elems = flat_out.numel()

    # Set a seed
    if generator is not None and hasattr(generator, "initial_seed"):
        seed_val = generator.initial_seed()
    else:
        seed_val = torch.randint(0, 2**31, ()).item()

    # Grid size
    BLOCK_SIZE = 1024
    grid = ( (n_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch
    fill_rand_kernel[grid](flat_out, seed_val, 0, n_elems, BLOCK_SIZE=BLOCK_SIZE)

    return out
