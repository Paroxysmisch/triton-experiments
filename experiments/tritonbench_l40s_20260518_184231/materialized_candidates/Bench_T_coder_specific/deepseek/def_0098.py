import tritonclient.grpc as grpcclient
import numpy as np

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Convert inputs to numpy arrays
    input_np = np.array(input)
    other_np = np.array(other)

    # Create Triton client
    triton_client = grpcclient.InferenceServerClient(url='localhost:8001')

    # Define inputs and outputs
    inputs = [grpcclient.InferInput('INPUT', input_np.shape, np.float32),
              grpcclient.InferInput('OTHER', other_np.shape, np.float32)]
    outputs = [grpcclient.InferRequestedOutput('OUTPUT')]

    # Set inputs
    inputs[0].set_data_from_numpy(input_np)
    inputs[1].set_data_from_numpy(other_np)

    # Set other parameters
    parameters = {'alpha': alpha, 'approximate': approximate}

    # Run inference
    response = triton_client.infer('sub_gelu', inputs, outputs=outputs, parameters=parameters)

    # Get output
    output_data = response.as_numpy('OUTPUT')

    return output_data
