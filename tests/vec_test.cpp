#include <gtest/gtest.h>
#include "main.cpp" // Include your main file or create a header for OnnxRVC class

class OnnxRVCTest : public ::testing::Test
{
  protected:
    std::unique_ptr<OnnxRVC> rvc;

    void SetUp() override
    {
        // Initialize OnnxRVC with test model paths
        rvc = std::make_unique<OnnxRVC>("/path/to/test/rvc_model.onnx",
                                        16000,
                                        512,
                                        "/path/to/test/vec_model.onnx");
    }

    void TearDown() override { rvc->cleanup(); }
};

TEST_F(OnnxRVCTest, ForwardVecModelBasicFunctionality)
{
    // Create a simple input vector
    std::vector<float> input(16000, 0.5f); // 1 second of audio at 16kHz

    // Call the forward_vec_model function
    Ort::Value output = rvc->forward_vec_model(input);

    // Check the output tensor shape
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[2], 768);

    // The middle dimension should be twice the input length divided by 320
    // (because of the time dimension doubling)
    EXPECT_EQ(output_shape[1], 2 * (16000 / 320));

    // Check that all values are within the expected range (-1 to 1)
    float* output_data     = output.GetTensorMutableData<float>();
    int64_t total_elements = std::accumulate(output_shape.begin(),
                                             output_shape.end(),
                                             1LL,
                                             std::multiplies<int64_t>());
    for (int64_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(output_data[i], -1.0f);
        EXPECT_LE(output_data[i], 1.0f);
    }
}

TEST_F(OnnxRVCTest, ForwardVecModelHandlesEmptyInput)
{
    std::vector<float> empty_input;

    // Expect an exception or specific behavior for empty input
    EXPECT_THROW(rvc->forward_vec_model(empty_input), std::runtime_error);
}

TEST_F(OnnxRVCTest, ForwardVecModelHandlesLargeInput)
{
    // Create a large input vector (e.g., 1 minute of audio)
    std::vector<float> large_input(16000 * 60, 0.5f);

    // Call the forward_vec_model function
    Ort::Value output = rvc->forward_vec_model(large_input);

    // Check the output tensor shape
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[2], 768);
    EXPECT_EQ(output_shape[1], 2 * (16000 * 60 / 320));
}

// Add more tests as needed

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}