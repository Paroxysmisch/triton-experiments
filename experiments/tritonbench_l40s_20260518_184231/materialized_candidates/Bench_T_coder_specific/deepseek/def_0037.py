import triton
import torch

@triton.jit
def _cosine_embedding_loss_kernel(input1_ptr, input2_ptr, target_ptr, output_ptr, num_elements, margin, reduction):
    # Implement the cosine embedding loss calculation here.
    pass

def fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=0, reduction='mean'):
    # Normalize inputs
    input1 = torch.nn.functional.normalize(input1, p=2, dim=1)
    input2 = torch.nn.functional.normalize(input2, p=2, dim=1)

    # Convert inputs and target to contiguous tensors
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    target = target.contiguous()

    # Allocate output tensor
    output = torch.empty((1,), dtype=input1.dtype, device=input1.device)

    # Call Triton kernel
    _cosine_embedding_loss_kernel[1, 1](
        input1.data_ptr(),
        input2.data_ptr(),
        target.data_ptr(),
        output.data_ptr(),
        input1.numel(),
        margin,
        reduction
    )

    return output
