#include <torch/torch.h>

int main(int argc, char const *argv[])
{
    /* code */
    torch::Tensor t = torch::eye(3);
    std::cout << t << "\n";

    return 0;
}
