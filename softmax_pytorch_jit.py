import torch

from softmax import naive_softmax

DEVICE = triton.runtime.driver.active.get_active_torch_device()

compiled_softmax = torch.compile(naive_softmax)

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_torch = compiled_softmax(x)
