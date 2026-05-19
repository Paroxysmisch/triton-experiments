import copy
import torch
import torch.nn as nn
from typing import Union, Dict, Set, Type

def quantize_dynamic(
    model: nn.Module,
    qconfig_spec: Union[Dict[Union[Type[nn.Module], str], Dict], Set[Union[Type[nn.Module], str]]] = None,
    inplace: bool = False,
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = None,
) -> nn.Module:
    """
    Converts a float model to a dynamic quantized model by replacing specified modules with dynamically quantized versions.

    Args:
        model: Input float model to be quantized.
        qconfig_spec: Configuration specifying which modules to quantize (either a set of types/names or a dict mapping types/names to qconfigs).
        inplace: If True, modifies the model in-place; otherwise, returns a copy.
        mapping: Custom mapping from module types to their dynamically quantized counterparts.

    Returns:
        The dynamically quantized model.
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Default mapping includes common layers like Linear and RNNs
    if mapping is None:
        from .quantized_modules import DynamicQuantizedLinear, DynamicQuantizedLSTM, DynamicQuantizedGRU  # Assume these are defined elsewhere
        mapping = {
            nn.Linear: DynamicQuantizedLinear,
            nn.LSTM: DynamicQuantizedLSTM,
            nn.GRU: DynamicQuantizedGRU,
        }
    
    # Determine which modules to replace and their qconfigs
    replacements = []
    for name, module in model.named_modules():
        qconfig = None
        if qconfig_spec is None:
            # Default behavior: quantize Linear and RNN modules
            if isinstance(module, tuple(mapping.keys())):
                qconfig = {'dtype': torch.qint8}  # Default to qint8
        else:
            if isinstance(qconfig_spec, set):
                if type(module) in qconfig_spec or name in qconfig_spec:
                    qconfig = {'dtype': torch.qint8}
            elif isinstance(qconfig_spec, dict):
                # Check for module type or name in qconfig_spec keys
                if type(module) in qconfig_spec:
                    qconfig = qconfig_spec[type(module)]
                elif name in qconfig_spec:
                    qconfig = qconfig_spec[name]
        
        if qconfig is not None:
            QuantizedModule = mapping.get(type(module))
            if QuantizedModule is not None:
                # Split the full name into parent and child
                if '.' in name:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                replacements.append((parent, child_name, module, QuantizedModule, qconfig))
    
    # Replace modules
    for parent, child_name, module, QuantizedModule, qconfig in replacements:
        quantized_module = QuantizedModule.from_float(module, qconfig)
        setattr(parent, child_name, quantized_module)
    
    return model
