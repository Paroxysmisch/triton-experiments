import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


def wrap_sub(G, name, arg_names, reduction_arg_names, kwargs, attrs):
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 1

    @reduction(
        size_hints=[
            instance_descriptor(arg_names[0]).size(0),
            instance_descriptor(arg_names[1]).size(0),
        ],
        reduction_hint=ReductionHint.INNER,
    )
    @triton.jit
    def fn(**META):
        N = META["N"]
        input_types = META["input_types"]
        output_type = META["output_type"]
        if input_types[0] is tl.int8:
            input_types[0] = tl.bfloat16
        if input_types[1] is tl.int8:
            input_types[1] = tl.bfloat16
        if len(input_types) == 3 and input_types[2] is tl.int8:
            input_types[2] = tl.bfloat16
        tl_dtypes = triton_helpers.promote_types_to_supported_by_device(input_types)
        tl_inputs, tl_outputs = triton_helpers.unpack_args(
            tl_dtypes,
            arg_names,
            kwargs,
            META["constants"],
            [META["ptr0"]],
            [META["ptr0"] + N],
            META["ptr1"],
            META["ptr1"] + N,
            META["ptr2"],
            META["ptr2"] + N if len(META["ptr2"]) > 0 else None,
        )
        if len(tl_inputs) == 3:
            tl_inputs[0].to(output_type, inplace=True)
            tl.store(tl_outputs, tl_inputs[0] - alpha * tl_inputs[1], mask=tl_inputs[2])
        else:
            tl_inputs[0].to(output_type, inplace=True)
            tl.store(tl_outputs, tl_inputs[0] - alpha * tl_inputs[1])

    arg_names_str = ",".join(arg_names)
    reduction_arg_names_str = ",".join(reduction_arg_names)
    G.add_function(
        f"{name}_reduction",
        fn,
        arg_names=arg_names_str,
        reduction_arg_names=reduction_arg_names_str,
        kwargs=kwargs,
        attrs=attrs,
    )
    G.add_primitive(name, torch.sub, attrs=attrs)


def func_inputs_to_wrapper_sub(func_inputs):
    """
    Convert func_inputs to a form that can be used to generate a Triton wrapper.
    """
    assert len(func_inputs) == 5
    assert func_inputs[0] == "G"
    assert func_inputs[1] == "name"
    assert isinstance(func_inputs[2], list)
    assert isinstance(func_inputs[3], list)
    assert isinstance(func_inputs[4], dict)
    return func_inputs
