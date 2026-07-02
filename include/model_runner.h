#pragma once
#include <torch/script.h>
#include <string>

class ModelRunner
{
public:
    explicit ModelRunner(const std::string &path);
    torch::Tensor forward(const torch::Tensor input);

private:
    torch::jit::script::Module module_;
};