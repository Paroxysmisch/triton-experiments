import tritonclient.grpc as grpcclient
import numpy as np

def exp_sqrt(input, out=None):
    try:
        if out is None:
            out = input.copy()

        # Connect to Triton server
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

        # Check if model is already loaded
        if not triton_client.is_model_ready("exp_sqrt"):
            # Load model
            triton_client.load_model("exp_sqrt")

        # Prepare input
        input_data = np.array([input], dtype=np.float32)

        # Prepare output
        output_data = np.empty(input.shape, dtype=np.float32)

        # Run inference
        response = triton_client.infer("exp_sqrt", inputs=[grpcclient.InferInput("INPUT", input_data.shape, "FP32"),], outputs=[grpcclient.InferRequestedOutput("OUTPUT", binary_data=False)])

        # Extract output
        output_data = np.array(response.as_numpy("OUTPUT"))

        return output_data

    except Exception as e:
        print("Exception: " + str(e))
        return None
