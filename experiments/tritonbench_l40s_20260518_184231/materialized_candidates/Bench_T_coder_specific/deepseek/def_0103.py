import torch
import torch.distributions as dist

def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor:
    # Compute bitwise AND
    result = input & other
    
    # Check if either probs or logits is provided
    if probs is not None:
        # If probs is provided, create a Binomial distribution with probs
        distribution = dist.Binomial(total_count, probs=probs)
    elif logits is not None:
        # If logits is provided, create a Binomial distribution with logits
        distribution = dist.Binomial(total_count, logits=logits)
    else:
        raise ValueError("Either probs or logits must be provided")

    # Sample from the distribution
    sample = distribution.sample()
    
    return sample
