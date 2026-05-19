import torch
from torch.quantization import default_dynamic_qconfig
from .quantize import _is_weight_only_quantizable

def quantize_dynamic(
    model: torch._C._dynamo.guards.GuardsT,
    qconfig_spec: torch._C._dynamo.guards.GuardsT = None,
    inplace: torch._C._dynamo.guards.GuardsT = False,
    mapping: torch._C._dynamo.guards.GuardsT = None,
) -> torch._C._dynamo.guards.GuardsT:
    """
    Converts a float model to a dynamic quantized model, optionally replacing specified modules with their dynamic weight-only quantized versions.

    Args:
        model (Model): The input model to quantize.
        qconfig_spec (Union[Dict[str, QConfig], Set[str]], optional): Specification for quantization configurations. Defaults to None.
        inplace (bool, optional): Whether to perform the quantization in place. Defaults to False.
        mapping (Dict[Type, Type], optional): A dictionary mapping module types to their dynamic weight-only quantized counterparts. Defaults to None.

    Returns:
        Model: The quantized model.
    """
    from torch.quantization import quantize_dynamic as _quantize_dynamic

    return _quantize_dynamic(
        model, qconfig_spec, inplace, mapping=mapping or _dynamic_wo_qconfig_mapping()
    )

def _dynamic_wo_qconfig_mapping() -> dict:
    from torch.ao.quantization import (
        DynamicQuantLinear,
        DynamicQuantEmbedding,
        DynamicQuantLSTM,
        DynamicQuantGRU,
        DynamicQuantRNN,
    )

    return {
        torch.nn.Linear: DynamicQuantLinear,
        torch.nn.Embedding: DynamicQuantEmbedding,
        torch.nn.LSTM: DynamicQuantLSTM,
        torch.nn.GRU: DynamicQuantGRU,
        torch.nn.RNN: DynamicQuantRNN,
    }
