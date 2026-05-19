import triton
from triton.language import *

@triton.autotune(
    configs=[
        AutoTuneConfig(grid=[1], block=[512]),
        AutoTuneConfig(grid=[1], block=[256]),
        AutoTuneConfig(grid=[1], block=[128]),
        AutoTuneConfig(grid=[1], block=[64]),
    ],
    key=['num_params']
)
def sgd(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, num_params=None):
    assert num_params is not None, "num_params must be specified"
    
    # Allocate memory for momentum buffer
    momentum_buffer = torch.empty_like(params, device=params.device)

    # Launch the kernel
    sgd_kernel[None](params, params.grad, momentum_buffer, lr, momentum, weight_decay, dampening, nesterov, maximize, num_params)

# Example usage
params = torch.randn(1024, requires_grad=True, device='cuda')
optimizer = lambda params, grad: sgd(params, grad, lr=0.01, momentum=0.9, weight_decay=1e-4, num_params=len(params))
optimizer(params, params.grad)
print(params)
