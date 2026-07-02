#include <torch/torch.h>
#include "tensor_io.h"

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        return 1;
    }
    torch::Tensor t = load_bin(argv[1], {1, 3, 108, 192});
    std::cout << "shape: " << t.sizes() << "\n";
    std::cout << "mean:  " << t.mean().item<float>() << "\n";
    std::cout << "min:   " << t.min().item<float>() << "\n";
    std::cout << "max:   " << t.max().item<float>() << "\n";

    return 0;
}
