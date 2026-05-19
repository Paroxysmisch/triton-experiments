import copy
import torch.nn as nn

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    if not inplace:
        model = copy.deepcopy(model)
    
    # Default mapping for common dynamic quantizable modules
    if mapping is None:
        from torch.quantization import DynamicQuantLinear, DynamicQuantLSTM, DynamicQuantGRU
        mapping = {
            nn.Linear: DynamicQuantLinear,
            nn.LSTM: DynamicQuantLSTM,
            nn.GRU: DynamicQuantGRU,
        }
    
    # Default qconfig_spec is the set of types in the mapping
    if qconfig_spec is None:
        qconfig_spec = set(mapping.keys())
    
    # Iterate through all modules to apply quantization
    for name, module in model.named_modules():
        quantize = False
        qconfig = None
        
        # Determine if the current module should be quantized
        if isinstance(qconfig_spec, set):
            if type(module) in qconfig_spec or name in qconfig_spec:
                quantize = True
        elif isinstance(qconfig_spec, dict):
            if type(module) in qconfig_spec:
                qconfig = qconfig_spec[type(module)]
                quantize = True
            elif name in qconfig_spec:
                qconfig = qconfig_spec[name]
                quantize = True
        else:
            raise TypeError("qconfig_spec must be a set or a dictionary")
        
        if quantize:
            quantized_cls = mapping.get(type(module))
            if quantized_cls is None:
                continue
            
            # Replace the module with its quantized version
            parent, child_name = _get_parent_child(model, name)
            if parent is not None:
                quantized_module = quantized_cls.from_float(module, qconfig)
                setattr(parent, child_name, quantized_module)
    
    return model

def _get_parent_child(model, full_name):
    if '.' not in full_name:
        return None, full_name  # Top-level module
    parent_name = '.'.join(full_name.split('.')[:-1])
    child_name = full_name.split('.')[-1]
    parent = model
    for part in parent_name.split('.'):
        if part:
            parent = getattr(parent, part)
    return parent, child_name
