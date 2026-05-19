import torch
from torch import qengine
from torch.ao.quantization import _quantize_weight, _get_default_qconfig

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    if qconfig_spec is None:
        qconfig_spec = {}

    if not inplace:
        model = model.clone(torch.ao.quantization.inplace.clone_qconfig)

    qconfigs = {}

    if isinstance(qconfig_spec, dict):
        for name, qconfig in qconfig_spec.items():
            if isinstance(name, str):
                module = model.get_submodule(name)
                if module is not None:
                    qconfigs[module] = qconfig
            elif isinstance(name, type):
                for module in model.submodules_of_type(name):
                    qconfigs[module] = qconfig
    else:
        for name in qconfig_spec:
            if isinstance(name, str):
                module = model.get_submodule(name)
                if module is not None:
                    qconfigs[module] = _get_default_qconfig(module)
            elif isinstance(name, type):
                for module in model.submodules_of_type(name):
                    qconfigs[module] = _get_default_qconfig(module)

    for module, qconfig in qconfigs.items():
        if isinstance(qconfig, dict):
            _quantize_weight(module, qconfig)
        else:
            _quantize_weight(module, {"dtype": qconfig})

    if mapping is not None:
        for name, new_cls in mapping.items():
            for module in model.named_modules_of_type(name):
                if isinstance(name, type):
                    module.replace_module_with_quantized(new_cls)

    return model
