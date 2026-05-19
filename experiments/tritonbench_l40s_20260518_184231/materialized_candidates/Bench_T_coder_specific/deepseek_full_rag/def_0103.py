import torch
import triton
import triton.language as tl

@triton.jit
def bitwise_and(x, y):
    return x & y

@triton.jit
def binomial(x, total_count, probs=None, logits=None):
    return tl.random.binomial(x, total_count, probs, logits)

def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor:
    input_and_other = bitwise_and(input, other)
    return binomial(input_and_other, total_count, probs, logits)
