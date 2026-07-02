#include <torch/torch.h>
#include "tensor_io.h"
#include "model_runner.h"
#include "preprocess.h"

int main(int argc, char const *argv[])
{
    if (argc != 5)
    {
        return 1;
    }
    torch::Tensor image_data = load_bin(argv[1], {1, 3, 108, 192});
    torch::Tensor joint_positions = load_bin(argv[2], {1, 6});

    ModelRunner cnn_model_runner(argv[3]);
    torch::Tensor vision_features = cnn_model_runner.forward(image_data);

    ModelRunner policy_model_runner(argv[4]);
    torch::Tensor observation = assemble_obs(vision_features, joint_positions);
    torch::Tensor action = policy_model_runner.forward(observation);

    std::cout << "shape: " << action.sizes() << "\n";
    std::cout << action << "\n";

    return 0;
}
