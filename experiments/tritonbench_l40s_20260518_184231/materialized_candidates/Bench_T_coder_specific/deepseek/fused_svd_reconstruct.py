import tritonclient.grpc as grpcclient
import numpy as np

def fused_svd_reconstruct(A: np.ndarray) -> np.ndarray:
    # Create Triton client
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Prepare inputs
    inputs = []
    inputs.append(grpcclient.InferInput("A", A.shape, "FP32"))
    inputs[0].set_data_from_numpy(A)

    # Run model
    outputs = triton_client.infer("fused_svd_reconstruct", inputs)

    # Get outputs
    A_reconstructed = outputs.as_numpy("A_reconstructed")

    return A_reconstructed
