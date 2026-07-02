#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>

// read a raw little-endian float32 file into a tensor of the given shape.
torch::Tensor load_bin(const std::string &path, at::IntArrayRef shape);