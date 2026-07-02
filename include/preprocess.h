#pragma once
#include <torch/torch.h>
#include <cstdint>

// Assemble the 1030-D observation from vision features and joint states
torch::Tensor assemble_obs(const torch::Tensor &vision_features, const torch::Tensor &joint_positions);