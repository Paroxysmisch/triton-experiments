import tritonclient.http as http_client
import numpy as np
from tritonclient.utils import InferenceServerException

def sqrt_tanh(input, out=None):
    # Convert input to NumPy array if it's not already
    input_np = np.array(input, dtype=np.float32)

    # Create a Triton client
    triton_client = http_client.InferenceServerClient(url="localhost:8000")

    # Prepare the input data
    input_tensor = http_client.InferInput("INPUT", input_np.shape, "FP32")
    input_tensor.set_data_from_numpy(input_np)

    # Prepare the output data
    if out is None:
        out = np.empty_like(input_np)
    output_tensor = http_client.InferRequestedOutput("OUTPUT", binary_data=False)

    # Perform inference
    try:
        results = triton_client.infer(model_name="sqrt_tanh_model",
                                       inputs=[input_tensor],
                                       outputs=[output_tensor])
    except InferenceServerException as e:
        print(e)
        return None

    # Get the output tensor
    output_np = results.as_numpy("OUTPUT")

    return output_np
