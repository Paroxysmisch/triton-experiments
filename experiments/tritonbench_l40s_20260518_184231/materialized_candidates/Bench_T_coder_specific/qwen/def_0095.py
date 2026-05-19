import tritonclient.http as http_client

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Initialize Triton client
    triton_client = http_client.InferenceServerClient(url="localhost:8000")

    # Create input tensors
    input_tensor = http_client.InferInput("input", input.shape, "FP32")
    input_tensor.set_data_from_numpy(input)

    running_mean_tensor = http_client.InferInput("running_mean", running_mean.shape, "FP32")
    running_mean_tensor.set_data_from_numpy(running_mean)

    running_var_tensor = http_client.InferInput("running_var", running_var.shape, "FP32")
    running_var_tensor.set_data_from_numpy(running_var)

    weight_tensor = None
    if weight is not None:
        weight_tensor = http_client.InferInput("weight", weight.shape, "FP32")
        weight_tensor.set_data_from_numpy(weight)

    bias_tensor = None
    if bias is not None:
        bias_tensor = http_client.InferInput("bias", bias.shape, "FP32")
        bias_tensor.set_data_from_numpy(bias)

    # Prepare outputs
    output_tensor = http_client.InferRequestedOutput("output")

    # Run inference
    results = triton_client.infer(model_name="custom_batch_norm",
                                 inputs=[input_tensor, running_mean_tensor, running_var_tensor, weight_tensor, bias_tensor],
                                 outputs=[output_tensor])

    # Get output tensor
    output = results.as_numpy("output")

    return output
