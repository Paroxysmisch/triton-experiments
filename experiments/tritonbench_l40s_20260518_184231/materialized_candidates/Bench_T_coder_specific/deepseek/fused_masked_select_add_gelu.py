import torch
import torch.nn.functional as F

def fused_masked_select_add_gelu(input, mask, other, alpha=1, approximate='none', out=None):
    # Masked selection
    masked_input = torch.masked_select(input, mask)
    
    # Addition
    added = masked_input + alpha * other
    
    # GELU activation
    if approximate == 'tanh':
        gelu_approx = F.gelu(added, approximate='tanh')
    else:
        gelu_approx = F.gelu(added)
    
    # Return the result
    return gelu_approx
