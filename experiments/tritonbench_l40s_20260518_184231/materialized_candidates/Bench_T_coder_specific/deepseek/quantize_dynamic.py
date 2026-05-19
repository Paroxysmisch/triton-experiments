def quantize_dynamic(model, qconfig_spec=None, inplace=False, mapping=None):
    # validate inputs
    if not isinstance(model, Model):
        raise TypeError("Expected model to be of type Model")
    if not isinstance(inplace, bool):
        raise TypeError("Expected inplace to be of type bool")
    if qconfig_spec is not None and not isinstance(qconfig_spec, (dict, set)):
        raise TypeError("Expected qconfig_spec to be of type dict or set")
    if mapping is not None and not isinstance(mapping, dict):
        raise TypeError("Expected mapping to be of type dict")

    # perform dynamic quantization
    if inplace:
        model.quantize_dynamic(qconfig_spec=qconfig_spec, mapping=mapping)
    else:
        model_copy = copy.deepcopy(model)
        model_copy.quantize_dynamic(qconfig_spec=qconfig_spec, mapping=mapping)
        return model_copy

    return model
