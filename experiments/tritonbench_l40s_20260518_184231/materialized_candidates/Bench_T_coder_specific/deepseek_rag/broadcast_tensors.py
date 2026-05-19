import triton
import torch

def sparse_dense_mm(sparse_tensor: torch.Tensor, dense_tensor: torch.Tensor):
    # Convert sparse tensor to BSR format
    sparse_tensor = sparse_tensor.to_sparse_bsr()
    
    # Call the Triton kernel with the converted sparse tensor and the dense tensor
    result = triton.language.dispatch.sampled_addmm(sparse_tensor, dense_tensor)
    
    return result
