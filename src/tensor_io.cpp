#include "tensor_io.h"
#include <iostream>
#include <fstream>

torch::Tensor load_bin(const std::string &path, at::IntArrayRef shape)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // get file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // compute number of floats
    // verify data is right sized for floats
    if (file_size % sizeof(float) != 0)
    {
        throw std::runtime_error("File size is not a multiple of float32: " + path);
    }

    // allocate
    size_t data_size = file_size / sizeof(float);
    std::vector<float> buffer(data_size);

    // read into buffer
    if (file_size > 0)
    {
        file.read(reinterpret_cast<char *>(buffer.data()), file_size);
    }
    if (!file)
        throw std::runtime_error("Read failed: " + path);

    torch::Tensor result = torch::from_blob(buffer.data(), shape, torch::kFloat32).clone();
    return result;
}