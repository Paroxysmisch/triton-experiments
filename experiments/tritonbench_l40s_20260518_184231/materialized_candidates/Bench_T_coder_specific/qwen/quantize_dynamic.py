import torch
from torch.quantization import dynamic_quantize_qconfig, default_dynamic_qconfig

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    """
    Convert a float model to a dynamic quantized model by replacing specified modules with their dynamic weight-only quantized versions.

    Args:
        model: input model
        qconfig_spec: Either a dictionary mapping submodule names/types to quantization configurations or a set of types/names for dynamic quantization
        inplace: carry out model transformations in-place, mutating the original module
        mapping: maps submodule types to dynamically quantized versions
    """
    if qconfig_spec is None:
        qconfig_spec = default_dynamic_qconfig('fbgemm')

    # Apply dynamic quantization
    if isinstance(qconfig_spec, dict):
        for name, qconfig in qconfig_spec.items():
            if name in model.state_dict():
                setattr(model, name, dynamic_quantize_qconfig(getattr(model, name), qconfig))
    elif isinstance(qconfig_spec, set):
        for module_name in qconfig_spec:
            for name, module in model.named_modules():
                if name.endswith(module_name):
                    setattr(model, name, dynamic_quantize_qconfig(module, qconfig_spec))

    # Inplace transformation
    if inplace:
        return model

    # Return new model
    return model.clone()

# Example usage
model = ...  # Your model here
qconfig_spec = {'Linear': torch.quantization.get_default_qconfig('fbgemm')}
quantized_model = quantize_dynamic(model, qconfig_spec=qconfig_spec, inplace=True)
