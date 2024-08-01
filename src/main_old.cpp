#include <onnxruntime_cxx_api.h>
#include "../lib/tinywav/myk_tiny.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <samplerate.h>

#include <world/harvest.h>
#include <world/dio.h>

// Downsampling function
std::vector<float>
resampleAudio(const std::vector<float>& inputAudio,
              int inputSampleRate,
              int outputSampleRate = 16000)
{
    if (inputSampleRate == outputSampleRate)
    {
        return inputAudio; // No need to downsample
    }

    // Calculate the required output size
    double ratio      = static_cast<double>(outputSampleRate) / inputSampleRate;
    size_t outputSize = static_cast<size_t>(inputAudio.size() * ratio);

    // Prepare the output buffer
    std::vector<float> outputAudio(outputSize);

    // Set up the SRC_DATA structure
    SRC_DATA srcData;
    srcData.data_in       = inputAudio.data();
    srcData.input_frames  = inputAudio.size();
    srcData.data_out      = outputAudio.data();
    srcData.output_frames = outputSize;
    srcData.src_ratio     = ratio;
    srcData.end_of_input  = 0;

    // Perform the downsampling
    int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
    if (error)
    {
        throw std::runtime_error(src_strerror(error));
    }

    return outputAudio;
}

std::pair<std::vector<float>, std::vector<float>>
estimateF0(const std::vector<double>& wav, int fs, int f0_method)
{
    HarvestOption option;
    InitializeHarvestOption(&option);

    // You can adjust these parameters if needed
    option.f0_floor     = 50.0;   // Hz
    option.f0_ceil      = 1100.0; // Hz
    option.frame_period = 10.0;   // ms

    int f0_length = GetSamplesForHarvest(fs, wav.size(), option.frame_period);
    std::vector<double> f0(f0_length);
    std::vector<double> temporal_positions(f0_length);

    if (f0_method == 0)
    { // Harvest
        Harvest(wav.data(),
                wav.size(),
                fs,
                &option,
                temporal_positions.data(),
                f0.data());
    }
    else
    { // Dio
      // If you need Dio, you'll need to implement it separately
      // as it has a different function signature
    }

    // Recast the f0 and temporal_positions vectors to float
    std::vector<float> f0_float(f0.begin(), f0.end());
    std::vector<float> temporal_positions_float(temporal_positions.begin(),
                                                temporal_positions.end());

    return { f0_float, temporal_positions_float };
}

void
sanityCheckContentVec(const std::string& modelPath,
                      const std::vector<float>& audio,
                      int sampleRate)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ContentVecTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelPath.c_str(), session_options);

    // Prepare input tensor
    std::vector<int64_t> input_shape = { 1,
                                         1,
                                         static_cast<int64_t>(audio.size()) };
    Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,
                                      const_cast<float*>(audio.data()),
                                      audio.size(),
                                      input_shape.data(),
                                      input_shape.size());

    // Define input and output names
    const char* input_names[]  = { "source" };
    const char* output_names[] = { "embed" };

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                      input_names,
                                      &input_tensor,
                                      1,
                                      output_names,
                                      1);

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    // Get output shape
    auto output_shape =
      output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    // Print output shape and first few values
    std::cout << "Output shape: [";
    for (auto dim : output_shape)
    {
        std::cout << dim << ",";
    }
    std::cout << "]\n";

    std::cout << "First few output values:\n";
    for (int i = 0;
         i < std::min(10, static_cast<int>(output_shape[1] * output_shape[2]));
         ++i)
    {
        std::cout << floatarr[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<float>
runContentVec(const std::string& modelPath,
              const std::vector<float>& audio,
              int sampleRate)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ContentVecTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelPath.c_str(), session_options);

    // Prepare input tensor
    std::vector<int64_t> input_shape = { 1,
                                         1,
                                         static_cast<int64_t>(audio.size()) };
    Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,
                                      const_cast<float*>(audio.data()),
                                      audio.size(),
                                      input_shape.data(),
                                      input_shape.size());

    // Define input and output names
    const char* input_names[]  = { "source" };
    const char* output_names[] = { "embed" };

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                      input_names,
                                      &input_tensor,
                                      1,
                                      output_names,
                                      1);

    // Get pointer to output tensor float values
    float* hubert_data = output_tensors.front().GetTensorMutableData<float>();

    // Get output shape
    auto output_shape =
      output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    // Calculate the size of the output data
    size_t output_size = 1;
    for (auto dim : output_shape)
    {
        output_size *= dim;
    }

    // Create a vector to store the hubert data
    std::vector<float> hubert_output(hubert_data, hubert_data + output_size);

    return hubert_output;
}

int
main(int argc, char* argv[])
{
    std::cout << "Hello from main.cpp" << std::endl;

    const char* inputPath = "/Users/thomaspower/Developer/Koala/RVC_Test/"
                            "test_audio/174-50561-0000.wav";
    const char* modelPath =
      "/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
      "vec-768-layer-12.onnx";

    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RVCTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Load audio
        int inputSampleRate = 16000, inputChannels = 1;
        std::vector<float> audio = myk_tiny::loadWav(inputPath);
        std::vector<float> outSignal(audio.size(), 0.0f);
        if (audio.empty())
        {
            std::cerr << "Failed to load audio or audio is empty." << std::endl;
            return 1;
        }

        // Downsample audio for contentVec to 16kHz
        std::vector<float> downsampledAudio =
          downsampleAudio(audio, 44100, 16000);

        // Test Harvest F0 estimation
        std::vector<double> audio_double(audio.begin(), audio.end());
        auto [f0, temporal_positions] =
          estimateF0(audio_double, inputSampleRate, 0);

        // Ok next I need to convert the f0 values to like a mel scale
        double f0_min     = 50.0;
        double f0_max     = 1100.0;
        double f0_mel_min = 1127 * std::log(1 + f0_min / 700);
        double f0_mel_max = 1127 * std::log(1 + f0_max / 700);

        std::vector<int64_t> pitch;
        for (auto p : f0)
        {
            double f0_mel = 1127 * std::log(1 + p / 700);
            if (f0_mel > 0)
            {
                f0_mel =
                  (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1;
            }
            f0_mel = std::clamp(f0_mel, 1.0, 255.0);
            pitch.push_back(static_cast<int64_t>(std::rint(f0_mel)));
        }

        // HUBERT ============================
        std::vector<float> hubert_data =
          runContentVec(modelPath, downsampledAudio, inputSampleRate);
        std::vector<int64_t> output_shape = {
            1, static_cast<int64_t>(hubert_data.size()) / 768, 768
        };
        int64_t num_frames   = output_shape[1];
        int64_t num_channels = output_shape[2];

        std::vector<float> hubert_repeated;
        hubert_repeated.reserve(num_frames * num_channels * 2);
        for (int64_t i = 0; i < num_frames; ++i)
        {
            for (int64_t j = 0; j < num_channels; ++j)
            {
                hubert_repeated.push_back(hubert_data[i * num_channels + j]);
                hubert_repeated.push_back(hubert_data[i * num_channels + j]);
            }
        }

        std::vector<float> hubert_transposed(num_frames * num_channels * 2);
        for (int64_t i = 0; i < num_frames; ++i)
        {
            for (int64_t j = 0; j < num_channels * 2; ++j)
            {
                hubert_transposed[j * num_frames + i] =
                  hubert_repeated[i * num_channels * 2 + j];
            }
        }

        int64_t sid = 0;

        // Prepare input tensors for RVC model
        std::vector<int64_t> phone_shape = {
            1, static_cast<int64_t>(hubert_transposed.size() / 768), 768
        };
        std::vector<int64_t> phone_lengths_shape = { 1 };
        std::vector<int64_t> pitch_shape         = {
            1, static_cast<int64_t>(pitch.size())
        };
        std::vector<int64_t> pitchf_shape = { 1,
                                              static_cast<int64_t>(f0.size()) };
        std::vector<int64_t> ds_shape     = { 1 };
        std::vector<int64_t> rnd_shape    = {
            1, 192, phone_shape[1]
        }; // rnd_dynamic_axes_1 should match phone_dynamic_axes_1

        int64_t phone_length = phone_shape[1];
        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value phone_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          hubert_transposed.data(),
                                          hubert_transposed.size(),
                                          phone_shape.data(),
                                          phone_shape.size());
        Ort::Value phone_lengths_tensor =
          Ort::Value::CreateTensor<int64_t>(memory_info,
                                            &phone_length,
                                            1,
                                            phone_lengths_shape.data(),
                                            phone_lengths_shape.size());
        Ort::Value pitch_tensor =
          Ort::Value::CreateTensor<int64_t>(memory_info,
                                            pitch.data(),
                                            pitch.size(),
                                            pitch_shape.data(),
                                            pitch_shape.size());
        Ort::Value pitchf_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          f0.data(),
                                          f0.size(),
                                          pitchf_shape.data(),
                                          pitchf_shape.size());
        Ort::Value ds_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info, &sid, 1, ds_shape.data(), ds_shape.size());

        // Generate random data for rnd tensor
        std::vector<float> rnd(1 * 192 * phone_shape[1]);
        std::generate(rnd.begin(),
                      rnd.end(),
                      []() { return static_cast<float>(rand()) / RAND_MAX; });
        Ort::Value rnd_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          rnd.data(),
                                          rnd.size(),
                                          rnd_shape.data(),
                                          rnd_shape.size());

        // Load and run RVC model
        const char* rvc_model_path = "/Users/thomaspower/Developer/Koala/"
                                     "RVC_Test/onnx_models/jojo-model.onnx";
        Ort::Session rvc_session(env, rvc_model_path, session_options);

        const char* input_names[]  = { "phone", "phone_lengths",
                                       "pitch", "pitchf",
                                       "ds",    "rnd" };
        const char* output_names[] = { "audio" };

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(phone_tensor));
        input_tensors.push_back(std::move(phone_lengths_tensor));
        input_tensors.push_back(std::move(pitch_tensor));
        input_tensors.push_back(std::move(pitchf_tensor));
        input_tensors.push_back(std::move(ds_tensor));
        input_tensors.push_back(std::move(rnd_tensor));

        std::cout << "Input tensor shapes:" << std::endl;

        // phone tensor
        std::cout << "phone: [";
        for (auto dim : phone_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;

        // phone_lengths tensor
        std::cout << "phone_lengths: [";
        for (auto dim : phone_lengths_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;

        // pitch tensor
        std::cout << "pitch: [";
        for (auto dim : pitch_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;

        // pitchf tensor
        std::cout << "pitchf: [";
        for (auto dim : pitchf_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;

        // ds tensor
        std::cout << "ds: [";
        for (auto dim : ds_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;

        // rnd tensor
        std::cout << "rnd: [";
        for (auto dim : rnd_shape)
        {
            std::cout << dim << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "f0 size: " << f0.size() << std::endl;
        std::cout << "pitch size: " << pitch.size() << std::endl;
        std::cout << "hubert_data size: " << hubert_data.size() << std::endl;
        std::cout << "hubert_transposed size: " << hubert_transposed.size()
                  << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;

        // Get number of inputs
        size_t num_input_nodes = rvc_session.GetInputCount();
        std::cout << "Number of inputs: " << num_input_nodes << std::endl;

        // Print input node names and shapes
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            // Get input node names
            Ort::AllocatedStringPtr input_name_ptr =
              rvc_session.GetInputNameAllocated(i, allocator);
            std::string input_name = input_name_ptr.get();

            // Get input node types
            Ort::TypeInfo type_info = rvc_session.GetInputTypeInfo(i);
            auto tensor_info        = type_info.GetTensorTypeAndShapeInfo();

            // Get input shapes/dims
            std::vector<int64_t> input_node_dims = tensor_info.GetShape();

            // Print
            std::cout << "Input " << i << " : name=" << input_name << std::endl;
            std::cout << "Input " << i << " : shape=[";
            for (size_t j = 0; j < input_node_dims.size(); j++)
            {
                if (input_node_dims[j] < 0)
                {
                    std::cout << "?";
                }
                else
                {
                    std::cout << input_node_dims[j];
                }
                if (j < input_node_dims.size() - 1)
                    std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }

        auto output_tensors = rvc_session.Run(Ort::RunOptions{ nullptr },
                                              input_names,
                                              input_tensors.data(),
                                              input_tensors.size(),
                                              output_names,
                                              1);
        // Post-process the output
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}