import torch

def fused_embedding_add_tanh(input_indices, weight, other, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    embeddings = torch.nn.functional.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    sum_tensor = embeddings + other
    output = torch.tanh(sum_tensor)
    return output
