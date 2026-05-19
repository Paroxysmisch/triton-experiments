import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic as torch_quantize_dynamic

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    """
    Converts a float model to a dynamic quantized model by replacing specified modules with their dynamic weight-only quantized versions.
    
    Args:
        model (torch.nn.Module): The input model to be quantized.
        qconfig_spec (dict or set, optional): Either a dictionary mapping submodule names/types to quantization configurations or a set of types/names for dynamic quantization.
        inplace (bool, optional): If True, carry out model transformations in-place, mutating the original module. Defaults to False.
        mapping (dict, optional): A dictionary that maps submodule types to dynamically quantized versions. Defaults to None.
    
    Returns:
        torch.nn.Module: The quantized model.
    """
    # Determine the dtype for quantization based on the qconfig_spec or default to float16
    dtype = torch.float16 if qconfig_spec is None else None
    
    # If qconfig_spec is provided, it overrides the dtype
    if qconfig_spec is not None:
        if isinstance(qconfig_spec, dict):
            # qconfig_spec is a dictionary mapping submodule names/types to quantization configurations
            qconfig_spec = {k: torch.quantization.default_dynamic_qconfig for k in qconfig_spec}
        elif isinstance(qconfig_spec, set):
            # qconfig_spec is a set specifying which submodules to apply dynamic quantization to
            qconfig_spec = {k: torch.quantization.default_dynamic_qconfig for k in qconfig_spec}
    
    # Perform the dynamic quantization
    quantized_model = torch_quantize_dynamic(
        model=model,
        qconfig_spec=qconfig_spec,
        dtype=dtype,
        inplace=inplace,
        mapping=mapping
    )
    
    return quantized_model
