#include "preprocess.h"

torch::Tensor assemble_obs(const torch::Tensor &vision_features, const torch::Tensor &joint_positions)
{
    return torch::cat({vision_features, joint_positions}, 1);
}