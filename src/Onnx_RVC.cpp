#pragma once

#include "Onnx_RVC.h"

OnnxRVC::OnnxRVC(const std::string& model_path,
                 int sr,
                 const std::string& vec_path)
  : model_sample_rate(model_sample_rate)

  , rvc_env(ORT_LOGGING_LEVEL_WARNING, "RVC_OnnxRVC")
  , vec_env(ORT_LOGGING_LEVEL_VERBOSE, "VEC_OnnxRVC")
  , vec_session(nullptr)
  , rvc_session(nullptr)
  , memory_info(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                           OrtMemTypeCPU))
  , inputBuffer_(16000 * 2)
  , outputBuffer_(16000 * 2)
{
    // Setup Onnx session objects
    Ort::SessionOptions session_options;

    Ort::SessionOptions session_options_vec;

    try
    {
        vec_session = std::make_unique<Ort::Session>(
          vec_env, vec_path.c_str(), session_options_vec);
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Error loading vec model: " << e.what() << std::endl;
        throw;
    }

    try
    {
        rvc_session = std::make_unique<Ort::Session>(
          rvc_env, model_path.c_str(), session_options);
        std::cout << "RVC model loaded successfully" << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Error loading RVC model: " << e.what() << std::endl;
        throw;
    }
}

OnnxRVC::~OnnxRVC()
{
    cleanup();
}

std::vector<float>
OnnxRVC::inference(std::vector<float> audio_data, float input_sample_rate)
{

    // Process the audio to get the input data for RVC
    auto [pitchf, pitch]     = compute_f0(audio_data, 1.0f);
    Ort::Value hubert_tensor = forward_vec_model(audio_data);

    // TODO: See if we can use Faiss to do vector index matching with trained
    // data (accent)

    // how many frames of hubert/content vec do we get.
    auto hubert_shape = hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int hubert_length = hubert_shape[1];

    // resample the pitch and pitchf to match the hubert length
    pitchf = Utils::resample_vector(pitchf, hubert_length);
    pitch  = Utils::resample_vector(pitch, hubert_length);

    // dummy speaker id (usually one models)
    std::vector<int64_t> dummy_ds = { 1 };

    // forward pass through RVC model
    std::vector<float> chunk_output =
      forward_rvc_model(hubert_tensor, hubert_length, pitch, pitchf, dummy_ds);

    return chunk_output;
}

Ort::Value
OnnxRVC::forward_vec_model(const std::vector<float>& wav16k)
{
    // Prepare input tensor
    std::vector<int64_t> input_shape = { 1,
                                         1,
                                         static_cast<int64_t>(wav16k.size()) };
    Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(memory_info,
                                      const_cast<float*>(wav16k.data()),
                                      wav16k.size(),
                                      input_shape.data(),
                                      input_shape.size());

    // Define input and output names
    const char* input_names[]  = { "source" };
    const char* output_names[] = { "embed" };

    assert(vec_session != nullptr);

    // Run inference
    auto output_tensors = vec_session->Run(Ort::RunOptions{ nullptr },
                                           input_names,
                                           &input_tensor,
                                           1,
                                           output_names,
                                           1);

    // Get output tensor
    Ort::Value& output_tensor = output_tensors.front();
    auto output_shape  = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    float* output_data = output_tensor.GetTensorMutableData<float>();

    int64_t total_elements = 1;
    for (const auto& dim : output_shape)
    {
        total_elements *= dim;
    }

    // Clamp values between -1 and 1, replace non-finite values with 0
    for (int64_t i = 0; i < total_elements; ++i)
    {
        if (!std::isfinite(output_data[i]))
        {
            output_data[i] = 0.0f;
        }
        else
        {
            output_data[i] = std::clamp(output_data[i], -1.0f, 1.0f);
        }
    }

    // Calculate new dimensions (the python code did this)
    int64_t batch    = output_shape[0];
    int64_t time     = output_shape[1];
    int64_t channels = output_shape[2];

    // Create new tensor with doubled time dimension
    std::vector<int64_t> new_shape = { batch, time * 2, channels };
    std::vector<float> new_data(batch * time * 2 * channels);

    // Repeat the time dimension
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t t = 0; t < time; ++t)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                float value =
                  output_data[b * (time * channels) + t * channels + c];
                new_data[b * (time * 2 * channels) + (t * 2) * channels + c] =
                  value;
                new_data[b * (time * 2 * channels) + (t * 2 + 1) * channels +
                         c] = value;
            }
        }
    }

    Ort::Value new_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                            new_data.data(),
                                                            new_data.size(),
                                                            new_shape.data(),
                                                            new_shape.size());

    return new_tensor;
}

// Get the f0 and pitch from the audio, this is freq and mel coefficients
std::pair<std::vector<float>, std::vector<int64_t>>
OnnxRVC::compute_f0(const std::vector<float>& wav, float up_key)
{
    return F0Estimator::computeF0WithPitch(wav, up_key);
}

// Forward pass through the RVC model
std::vector<float>
OnnxRVC::forward_rvc_model(Ort::Value& hubert_tensor,
                           int hubert_length,
                           const std::vector<int64_t>& pitch,
                           const std::vector<float>& pitchf,
                           const std::vector<int64_t>& ds)
{

    float* hubert_data = hubert_tensor.GetTensorMutableData<float>();

    // Prepare input tensors
    std::vector<int64_t> hubert_shape = { 1, hubert_length, 768 };
    std::vector<int64_t> length_shape = { 1 };
    std::vector<int64_t> pitch_shape  = { 1, hubert_length };
    std::vector<int64_t> ds_shape     = { 1 };

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info,
      reinterpret_cast<int64_t*>(&hubert_length),
      1,
      length_shape.data(),
      length_shape.size());
    Ort::Value pitch_tensor =
      Ort::Value::CreateTensor<int64_t>(memory_info,
                                        const_cast<int64_t*>(pitch.data()),
                                        pitch.size(),
                                        pitch_shape.data(),
                                        pitch_shape.size());
    Ort::Value pitchf_tensor =
      Ort::Value::CreateTensor<float>(memory_info,
                                      const_cast<float*>(pitchf.data()),
                                      pitchf.size(),
                                      pitch_shape.data(),
                                      pitch_shape.size());
    Ort::Value ds_tensor =
      Ort::Value::CreateTensor<int64_t>(memory_info,
                                        const_cast<int64_t*>(ds.data()),
                                        ds.size(),
                                        ds_shape.data(),
                                        ds_shape.size());

    // Define input and output names
    const char* input_names[]  = { "feats", "p_len", "pitch", "pitchf", "sid" };
    const char* output_names[] = { "audio" };

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(5);
    input_tensors.push_back(std::move(hubert_tensor));
    input_tensors.push_back(std::move(length_tensor));
    input_tensors.push_back(std::move(pitch_tensor));
    input_tensors.push_back(std::move(pitchf_tensor));
    input_tensors.push_back(std::move(ds_tensor));

    // Run inference
    std::vector<Ort::Value> output_tensors;
    try
    {
        output_tensors = rvc_session->Run(Ort::RunOptions{ nullptr },
                                          input_names,
                                          input_tensors.data(),
                                          input_tensors.size(),
                                          output_names,
                                          1);
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error during inference: " << e.what()
                  << std::endl;
        throw;
    }

    // Process output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    size_t output_size =
      output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    // Convert to vector and squeeze
    std::vector<float> out_wav(output_data, output_data + output_size);
    if (out_wav.size() > 0)
    {
        out_wav.shrink_to_fit();
    }

    // Check for NaN values
    bool has_nan = false;
    for (const auto& value : out_wav)
    {
        if (std::isnan(value))
        {
            has_nan = true;
            break;
        }
    }

    if (has_nan)
    {
        std::cerr << "Warning: NaN values detected in the output." << std::endl;
    }

    return out_wav;
}

void
OnnxRVC::cleanup()
{
    {
        // Delete sessions
        if (vec_session)
        {
            vec_session.reset();
        }
        if (rvc_session)
        {
            rvc_session.reset();
        }

        // Clear any stored data
        wav16k.clear();
        pitchf.clear();
        pitch.clear();
        ds.clear();

        // Release any Ort::Value tensors
        if (hubert_tensor)
        {
            hubert_tensor = Ort::Value(nullptr);
        }

        // Force garbage collection
        Ort::AllocatorWithDefaultOptions allocator;
        allocator.GetInfo();
    }
}