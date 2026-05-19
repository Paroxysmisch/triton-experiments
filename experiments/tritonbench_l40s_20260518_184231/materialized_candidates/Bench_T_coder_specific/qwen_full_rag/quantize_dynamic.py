import torch
from .quant_utils import get_qconfig_mapping
from .qat import replace_qat_module
from .observer import register_observer


class _DynamicQuantModel(torch.nn.Module):
    def __init__(self, float_model, mapping, qconfig, scale_dtype):
        super().__init__()
        self.float_model = float_model
        self.mapping = mapping
        self.qconfig = qconfig
        self.scale_dtype = scale_dtype

    @torch.no_grad()
    def prepare_dynamic_scales(self):
        for name, module in self.float_model.named_modules():
            if type(module) in self.mapping.values():
                weight = getattr(module, "weight")
                scale = torch.empty(weight.shape[0], dtype=self.scale_dtype, device="cuda")
                setattr(module, "scale", scale)

    def forward(self, *args, **kwargs):
        return self.float_model(*args, **kwargs)


def quantize_dynamic(
    model: torch.nn.Module,
    qconfig_spec: dict = None,
    inplace: bool = False,
    dtype=torch.qint8,
    qconfig=None,
    backend="default",
) -> torch.nn.Module:
    """
    Convert a float model to a dynamic quantized model by replacing specified
    modules with their dynamic weight-only quantized versions.

    Simple Usage:

    .. code-block:: python

        from torch.quantization import quantize_dynamic
        quantized_model = quantize_dynamic(model, {Linear: WeightOnlyDynamicQuant})

    Fine-grained control:

    .. code-block:: python

        from torch.quantization import quantize_dynamic
        from torch.quantization.qconfig import get_default_qconfig
        qconfig = get_default_qconfig("fbgemm")
        quantized_model = quantize_dynamic(model, qconfig_spec=qconfig, inplace=True)

    Args:
        model (torch.nn.Module): A float model to convert to dynamic quantized model
        qconfig_spec (dict/set, optional): Specification of quantization configuration or target modules/types to apply dynamic quantization to. Defaults to None.
        inplace (bool, optional): Carry out model transformation in-place, mutating the original module. Defaults to False.
        dtype (:class:`torch.dtype`, optional): Quantized data type. Only used when qconfig is None. Defaults to torch.qint8.
        qconfig (:class:`QConfig`, optional): Quantization configuration. If provided, this overrides dtype. Defaults to None.
        backend (str, optional): Target backend. Currently unused. Defaults to "default".

    Returns:
        torch.nn.Module: Dynamic quantized model
    """

    class CustomReplaceFn:
        def __call__(self, mod):
            return replace_qat_module(mod, self.mapping, self.wrapper_fn)

    def add_dynamic_scales(observer, wrapper):
        observer.model.prepare_dynamic_scales()

    wrapper_fn = None
    if not inplace:
        wrapper_fn = partial(_DynamicQuantModel, qconfig=qconfig, scale_dtype=dtype)

    qconfig_mapping = get_qconfig_mapping(qconfig_spec, qconfig)
    register_observer(qconfig_mapping, add_dynamic_scales)

    quantized_model = quantize_dynamic_model(
        model,
        custom_replace_function=CustomReplaceFn(),
        qconfig_mapping=qconfig_mapping,
        wrapper_fn=wrapper_fn,
    )

    return quantized_model
