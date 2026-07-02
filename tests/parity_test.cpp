#include <gtest/gtest.h>
#include "model_runner.h"
#include "preprocess.h"
#include "tensor_io.h"

static void expect_close(const torch::Tensor &a, const torch::Tensor &b,
                         double atol = 1e-4, double rtol = 1e-4)
{
    ASSERT_EQ(a.sizes(), b.sizes());
    EXPECT_TRUE(torch::allclose(a, b, rtol, atol))
        << "max abs diff: " << (a - b).abs().max().item<float>();
}

TEST(Parity, EncoderStage)
{
    ModelRunner enc("models/encoder.pt");
    auto in = load_bin("fixtures/encoder_in.bin", {1, 3, 108, 192});
    auto ref = load_bin("fixtures/encoder_out.bin", {1, 1024});
    expect_close(enc.forward(in), ref);
}

TEST(Parity, PolicyStage)
{
    ModelRunner policy("models/policy.pt");
    auto in = load_bin("fixtures/policy_in.bin", {1, 1030});
    auto ref = load_bin("fixtures/policy_out.bin", {1, 6});
    expect_close(policy.forward(in), ref);
}

TEST(Parity, EndToEnd)
{
    auto enc_in = load_bin("fixtures/e2e_img.bin", {1, 3, 108, 192});
    auto joints_in = load_bin("fixtures/e2e_joints.bin", {1, 6});
    auto ref = load_bin("fixtures/e2e_action.bin", {1, 6});

    ModelRunner encoder("models/encoder.pt");
    ModelRunner policy("models/policy.pt");
    expect_close(policy.forward(assemble_obs(encoder.forward(enc_in), joints_in)), ref);
}