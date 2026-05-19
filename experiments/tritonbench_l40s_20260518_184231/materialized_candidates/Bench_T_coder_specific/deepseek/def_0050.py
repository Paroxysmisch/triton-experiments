import tritonclient.grpc as grpcclient
import numpy as np

def sqrt_exp(input, out=None):
    try:
        # Create a Triton client
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)

        # Define the model
        model_name = "sqrt_exp"
        model_version = "1"

        # Define the input
        inputs = []
        inputs.append(grpcclient.InferInput("INPUT", input.shape, "FP32"))
        inputs[0].set_data_from_numpy(input.numpy())

        # Define the output
        outputs = []
        if out is not None:
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT", class_names=None, binary_data=False))

        # Run the inference
        response = triton_client.infer(model_name, model_version, inputs, outputs=outputs)

        # Return the output
        return np.array(response.as_numpy("OUTPUT"))

    except Exception as e:
        print("An error occurred: ", str(e))
