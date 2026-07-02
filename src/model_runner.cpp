#include "model_runner.h"

ModelRunner::ModelRunner(const std::string &path)
{
    // load model from path, save to module_
    module_ = torch::jit::load(path);
    module_.eval();
}

torch::Tensor ModelRunner::forward(const torch::Tensor input)
{
    {
        torch::NoGradGuard no_grad;
        return module_.forward({input}).toTensor();
    }
}
