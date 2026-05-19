import torch
from torch.utils.cpp_extension import load

# Load the Triton operator
triton = load(name="triton", sources="path/to/your/triton_operator.cu")

# Define your input tensors
logits = torch.randn(10, 100, requires_grad=True, device='cuda')  # batch of 10 sequences, each with 100 tokens
b_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], device='cuda')  # start locations of each sequence
b_seqlen = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], device='cuda')  # length of each sequence

# Create an output tensor
prob_out = torch.empty_like(logits, device='cuda')

# Call the Triton operator
triton.token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out)

# Print the output
print(prob_out)
