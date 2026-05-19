import tritonclient.grpc as grpcclient
import numpy as np

def log_tanh(input):
    try:
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)

        inputs = []
        outputs = []

        input_tensor = grpcclient.InferInput('INPUT', input.shape, "FP32")
        input_tensor.set_data_from_numpy(input)
        inputs.append(input_tensor)

        output_tensor = grpcclient.InferRequestedOutput('OUTPUT')
        outputs.append(output_tensor)

        response = triton_client.infer('log_tanh', inputs, request_id='1', outputs=outputs)

        return response.as_numpy('OUTPUT')

    except Exception as e:
        print("Inference Request failed: " + str(e))
