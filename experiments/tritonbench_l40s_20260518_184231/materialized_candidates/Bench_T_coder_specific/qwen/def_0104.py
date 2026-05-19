import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import numpy as np

def rad2deg_sqrt(input_tensor):
    # Create a Triton client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")

    # Check if the model is ready
    model_ready = triton_client.is_model_ready("rad2deg_sqrt")
    if not model_ready:
        raise Exception("Model is not ready")

    # Prepare input data
    input_data = input_tensor.numpy()
    input_shape = input_data.shape
    input_dtype = input_data.dtype

    # Create input tensor descriptor
    input_tensor_descriptor = httpclient.InferInput(
        "input", input_shape, "FP32"
    )
    input_tensor_descriptor.set_data_from_numpy(input_data)

    # Create output tensor descriptors
    output_deg_descriptor = httpclient.InferRequestedOutput("output_deg")
    output_sqrt_descriptor = httpclient.InferRequestedOutput("output_sqrt")

    # Perform inference
    response = triton_client.infer(
        model_name="rad2deg_sqrt",
        inputs=[input_tensor_descriptor],
        outputs=[output_deg_descriptor, output_sqrt_descriptor]
    )

    # Extract output tensors
    output_deg = response.as_numpy("output_deg")
    output_sqrt = response.as_numpy("output_sqrt")

    return output_deg, output_sqrt
