import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:
    from torch.ao.quantization.qconfig import QConfigMapping
else:
    from torch.ao.quantization.qconfig import QConfigSpec

def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    if not TRITON3:
        assert isinstance(qconfig_spec, QConfigSpec), "qconfig_spec must be a QConfigSpec if torch version < 3.0.0"
    else:
        assert isinstance(qconfig_spec, QConfigMapping), "qconfig_spec must be a QConfigMapping if torch version >= 3.0.0"

    assert isinstance(inplace, bool), "inplace must be a boolean"

    if not inplace:
        model = model.copy()

    # replace specified modules with their dynamic quantized versions
    model = DynamicQuantize.apply(model, qconfig_spec, mapping)

    return model
