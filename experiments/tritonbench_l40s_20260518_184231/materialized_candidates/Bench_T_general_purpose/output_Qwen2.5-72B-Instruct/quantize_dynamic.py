import triton
import triton.language as tl

@triton.jit
def dynamic_quantize_kernel(
    weight_ptr,  # Pointer to the weight tensor
    quantized_weight_ptr,  # Pointer to the quantized weight tensor
    scale_ptr,  # Pointer to the scale tensor
    zero_point_ptr,  # Pointer to the zero point tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < weight_ptr.size

    weight = tl.load(weight_ptr + offsets, mask=mask)
    quantized_weight = tl.libdevice.convert_to_int8(weight, scale_ptr, zero_point_ptr)
    tl.store(quantized_weight_ptr + offsets, quantized_weight, mask=mask)

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig, QConfig, default_dynamic_qconfig
from torch.ao.quantization import quantize_dynamic as torch_quantize_dynamic

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None, dtype=torch.qint8):
    """
    Converts a float model to a dynamic quantized model by replacing specified modules with their dynamic weight-only quantized versions.
    
    Args:
        model (torch.nn.Module): Input model to be quantized.
        qconfig_spec (dict or set, optional): Either a dictionary mapping submodule names/types to quantization configurations or a set of types/names for dynamic quantization.
        inplace (bool, optional): If True, carry out model transformations in-place, mutating the original module. Default is False.
        mapping (dict, optional): Maps submodule types to dynamically quantized versions.
        dtype (torch.dtype, optional): The data type for quantization. Supported values are torch.float16 and torch.qint8. Default is torch.qint8.
    
    Returns:
        torch.nn.Module: The quantized model.
    """
    if qconfig_spec is None:
        qconfig_spec = default_dynamic_qconfig if dtype == torch.qint8 else None

    if mapping is None:
        mapping = {
            nn.Linear: nn.qat.Linear,
            nn.LSTM: nn.qat.LSTM,
            nn.GRU: nn.qat.GRU,
            nn.RNN: nn.qat.RNN,
        }

    if dtype == torch.float16:
        qconfig = QConfig(activation=None, weight=torch.ao.quantization.float16_dynamic_qconfig)
    elif dtype == torch.qint8:
        qconfig = QConfig(activation=None, weight=torch.ao.quantization.default_weight_only_qconfig)
    else:
        raise ValueError("Unsupported dtype. Supported values are torch.float16 and torch.qint8.")

    if qconfig_spec is not None:
        if isinstance(qconfig_spec, dict):
            qconfig_dict = qconfig_spec
        elif isinstance(qconfig_spec, set):
            qconfig_dict = {k: qconfig for k in qconfig_spec}
        else:
            raise ValueError("qconfig_spec must be a dictionary or a set.")
    else:
        qconfig_dict = {k: qconfig for k in mapping.keys()}

    quantized_model = torch_quantize_dynamic(
        model,
        qconfig_spec=qconfig_dict,
        inplace=inplace,
        mapping=mapping
    )

    return quantized_model
