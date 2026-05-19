import triton
import triton.language as tl
import torch
import copy
from typing import Optional, Dict, Set, Union, Type
from torch.nn import Module

# Helper function to get configurations for autotuning
def _get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512}, num_warps=16),
    ]

@triton.jit
def _dynamic_quantize_kernel(
    weight_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Kernel for dynamic quantization of weights"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE_M
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < n_elements

    # Load weights
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Compute max absolute value for scaling
    abs_weight = tl.abs(weight)
    max_abs = tl.max(abs_weight, axis=0)
    scale = max_abs / 127.0
    
    # Quantize weights
    quant_weight = tl.where(
        mask,
        tl.round(weight / scale).to(tl.int8),
        0
    )
    
    # Store results
    tl.store(output_ptr + offsets, quant_weight, mask=mask)
    if pid == 0:
        tl.store(scale_ptr, scale)

def quantize_weight_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to quantize a single weight tensor"""
    n_elements = weight.numel()
    output = torch.empty_like(weight, dtype=torch.int8)
    scale = torch.empty((), dtype=weight.dtype, device=weight.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE_M']),)
    
    _dynamic_quantize_kernel[grid](
        weight.data_ptr(),
        output.data_ptr(),
        scale.data_ptr(),
        n_elements,
        BLOCK_SIZE_M=128,
    )
    
    return output, scale

class DynamicallyQuantizedLinear(torch.nn.Module):
    """Dynamic quantized version of nn.Linear"""
    def __init__(self, linear_module: torch.nn.Linear):
        super().__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        
        # Quantize weights
        self.weight_int8, self.weight_scale = quantize_weight_tensor(
            linear_module.weight.data
        )
        
        if linear_module.bias is not None:
            self.bias = torch.nn.Parameter(linear_module.bias.data)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # Dequantize weights during forward pass
        weight_deq = self.weight_int8.float() * self.weight_scale
        return torch.nn.functional.linear(input, weight_deq, self.bias)

def quantize_dynamic(
    model: Module,
    qconfig_spec: Optional[Union[Dict, Set]] = None,
    inplace: bool = False,
    mapping: Optional[Dict[Type, Type]] = None
) -> Module:
    """
    Converts a float model to a dynamic quantized model.
    
    Args:
        model: input model
        qconfig_spec: Dictionary mapping submodule names/types to quantization 
                     configurations or a set of types/names for dynamic quantization
        inplace: If True, modify model in-place
        mapping: Dictionary mapping module types to their quantized versions
    
    Returns:
        Quantized version of the model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    if mapping is None:
        mapping = {
            torch.nn.Linear: DynamicallyQuantizedLinear,
            # Add other module types as needed
        }
    
    def _should_quantize(name: str, module: Module) -> bool:
        if qconfig_spec is None:
            return type(module) in mapping
        if isinstance(qconfig_spec, set):
            return name in qconfig_spec or type(module) in qconfig_spec
        return name in qconfig_spec or type(module) in qconfig_spec
    
    # Recursively quantize applicable modules
    for name, module in model.named_children():
        if _should_quantize(name, module):
            quantized_type = mapping.get(type(module))
            if quantized_type is not None:
                setattr(model, name, quantized_type(module))
        else:
            quantize_dynamic(module, qconfig_spec, True, mapping)
    
    return model
