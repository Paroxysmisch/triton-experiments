import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"DEPTH": 16}, num_warps=2),
        triton.Config({"DEPTH": 16}, num_warps=4),
        triton.Config({"DEPTH": 16}, num_warps=8),
        triton.Config({"DEPTH": 32}, num_warps=2),
        triton.Config({"DEPTH": 32}, num_warps=4),
        triton.Config({"DEPTH": 32}, num_warps=8),
        triton.Config({"DEPTH": 64}, num_warps=2),
        triton.Config({"DEPTH": 64}, num_warps=4),
        triton.Config({"DEPTH": 64}, num_warps=8),
        triton.Config({"DEPTH": 128}, num_warps=2),
        triton.Config({"DEPTH": 128}, num_warps=4),
        triton.Config({"DEPTH": 128}, num_warps=8),
        triton.Config({"DEPTH": 256}, num_warps=2),
        triton.Config({"DEPTH": 256}, num_warps=4),
        triton.Config({"DEPTH": 256}, num_warps=8),
    ],
    key=["V"],
)
@triton.heuristics({"IS_FP16": lambda args: args["T"] == tl.float16})
@triton.jit
def _softmax(
    OUT,
    IN,
    LOG: tl.constexpr,
    MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr,
    V: tl.constexpr,
    stride,  # how much to increase the pointer to advance 1 row
    depth,  # how much to increase the pointer to advance 1 column
    depth2,  # how much to increase the pointer to advance V columns
    mask_pointer,
    BLOCK_SIZE: tl.constexpr,
    DEPTH: tl.constexpr,
):
    # Map the program id to the row of `OUT` it should compute.
    row = tl.program_id(0)
    col = tl.program_id(1)
    # The memory address of all the elements that we want to load can be computed as follows
    in_pointer = IN + row * stride + col * depth
    out_pointer = OUT + row * stride + col * depth
    mask_pointer = mask_pointer + row * stride + col * depth

    # Load input data; pad out out-of-bounds elements with 0
    if CAUSAL:
        causal_mask = col * depth + tl.arange(0, DEPTH) < (col + 1) * depth
        input = tl.load(in_pointer + tl.arange(0, DEPTH), mask=causal_mask, other=-float("inf"))
    else:
        input = tl.load(in_pointer + tl.arange(0, DEPTH), mask=None, other=-float("inf"))
    if MASK:
        mask = tl.load(mask_pointer + tl.arange(0, DEPTH), mask=None, other=0)
        input = input + mask

    # Subtract maximum for numerical stability
    input_minus_max = input - tl.max(input, axis=0)
    # Note that `tl.exp` is fast but approximate (i.e., think __expf in CUDA)
    # If full precision is required, use `tl.exp2` instead
    numerator = tl.exp(input_minus_max)
    denominator = tl.sum(numerator, axis=0)
    if LOG:
        output = input_minus_max - tl.log(denominator)
    else:
        output = numerator / denominator

    # Write output
    tl.store(out_pointer + tl.arange(0, DEPTH), output, mask=None)

def softmax(inp, dim=None, mask=None, causal=False, log=False):
    if dim is None:
        dim = -1
    elif dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError("Dimension out of range")

    if mask is not None:
        if mask.ndim != inp.ndim:
            raise ValueError("The shape of mask must be broadcastable with the shape of the underlying tensor")
        mask = mask.contiguous()
    inp = inp.contiguous()

    # Normalize `dim` to be non-negative
    if dim < 0:
        dim = inp.ndim + dim

    # The kernel only works on the last dimension, so we reshape the input
    # tensor to move the dimension of interest to the back
    perm = [i for i in range(inp.ndim)]
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inp = inp.permute(perm)
    out = torch.empty_like(inp)

    V = triton.next_power_of_2(inp.shape[-1])
    grid = lambda meta: (
        triton.cdiv(inp.shape[-2], meta["BLOCK_SIZE"]),
        triton.cdiv(inp.shape[-3], meta["DEPTH"]),
    )
    _softmax[grid](
        out,
        inp,
        log,
        mask is not None,
        causal,
        mask is not None and not causal,
        V,
        stride=inp.stride(-3),
        depth=inp.stride(-2),
        depth2=inp.stride(-1),
        mask_pointer=mask.stride(-3) if mask is not None else 0,
        BLOCK_SIZE=inp.shape[-2],
    )
    # Now we permute the output tensor back to the original order
    out = out.permute(perm)
    return out

@triton.autotune(
    configs=[
        triton.Config({"DEPTH": 16}, num_warps=2),
        triton.Config({"DEPTH": 16}, num_warps=4),
        triton.Config({"DEPTH": 16}, num_warps=8),
        triton.Config({"DEPTH": 32}, num_warps=2),
        triton.Config({"DEPTH": 32}, num_warps=4),
        triton.Config({"DEPTH": 32}, num_warps=8),
        triton.Config({"DEPTH": 64}, num_warps=2),
        triton.Config({"DEPTH": 64}, num_warps=4),
        triton.Config({"DEPTH": 64}, num_warps=8),
        triton.Config({"DEPTH": 128}, num_warps=2),
        triton.Config({"DEPTH": 128}, num_warps=4),
        triton.Config({"DEPTH": 128}, num_warps=8),
        triton.Config({"DEPTH": 256}, num_warps=2),
        triton.Config({"DEPTH": 256}, num_warps=4),
        triton.Config({"DEPTH": 256}, num_warps=8),
    ],
    key=["V"],
)
@triton.heuristics({"IS_FP16": lambda args: args["T"] == tl.float16})
@triton.jit
def _softmax_backward(
    OUT,
    IN,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_MASK_TYPE: tl.constexpr,
    IS_FP16: tl.constexpr,
    V: tl.constexpr,
    stride,  # how much to increase the pointer to advance 1 row
    depth,  # how much to increase the pointer to advance 1 column
    depth2,  # how much to increase the pointer to advance V columns
    BLOCK_SIZE: tl.constexpr,
    DEPTH: tl.constexpr,
):
    # Map the program id to the row of `OUT` it should compute.
    row = tl.program_id(0)
    col = tl.program_id(1)
    # The memory address of all the elements that we want to load can be computed as follows
    in_pointer = IN + row * stride + col * depth
    out_pointer = OUT + row * stride + col * depth

    # Load input data; pad out of bounds elements with 0
    if CAUSAL:
        causal_mask = col * depth + tl.arange(0, DEPTH) < (col + 1) * depth
        out = tl.load(out_pointer + tl.arange(0, DEPTH), mask=causal_mask, other=0)
        input = tl.load(in_pointer + tl.arange(0, DEPTH), mask=causal_mask, other=-float("inf"))
    else:
        out = tl.load(out_pointer + tl.arange(0, DEPTH), mask=None, other=0)
        input = tl.load(in_pointer + tl.arange(0, DEPTH), mask=None, other=-float("inf"))
    input_minus_one = input - 1
    # Note that `tl.exp` is fast but approximate (i.e., think __expf in CUDA)
    # If full precision is required, use `tl.exp2` instead
    # Also, branch-free implementation using `tl.where` (equivalent to a piecewise function)
    numerator = tl.where(input <= 0, tl.exp(input), tl.exp2(input))
    denominator = tl.sum(numerator, axis=0)
    if LOG:
        output = out * (input_minus_one - tl.log(denominator))
    else:
        output = out * (input_minus_one - numerator / denominator)

    # Write output
    tl.store(in_pointer + tl.arange(0, DEPTH), output, mask=None)

def softmax_backward(inp, grad_output, dim=None, causal=False, log=False):
    if dim is
