cpp
#include <triton/api.h>

namespace tl = triton::language;

class MVOperator : public tl::UserOperator {
public:
    MVOperator() : UserOperator("mv") {
        this->addInput("A", tl::DataType::FLOAT32, {-1, -1});
        this->addInput("B", tl::DataType::FLOAT32, {-1});
        this->addOutput("C", tl::DataType::FLOAT32, {-1});
    }

    void execute(tl::ExecutionContext* context) {
        // Retrieve input tensors
        auto A = context->getInputTensor("A");
        auto B = context->getInputTensor("B");

        // Retrieve output tensor
        auto C = context->getOutputTensor("C");

        // Perform matrix-vector multiplication
        // ...

        // Synchronize device to ensure all operations have completed
        context->synchronize();
    }
};
