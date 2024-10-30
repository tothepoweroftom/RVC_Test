#pragma once
#include <onnxruntime_cxx_api.h>
#include "../lib/tinywav/myk_tiny.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <samplerate.h>
#include <sstream>
#include <memory>
#include <cassert>
#include <numeric>
#include <AudioFile.h>

#include <world/harvest.h>
#include <world/dio.h>

#include "F0Estimator.h"

static const int sr = 40000;

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

class OnnxRVC
{
  public:
    OnnxRVC(const std::string& model_path,
            int sr                      = 16000,
            int hop_size                = 512,
            const std::string& vec_path = "/path/to/vec-768-layer-12.onnx")
      : sampling_rate(sr)
      , hop_size(hop_size)
      , rvc_env(ORT_LOGGING_LEVEL_WARNING, "RVC_OnnxRVC")
      , vec_env(ORT_LOGGING_LEVEL_VERBOSE, "VEC_OnnxRVC")
      , vec_session(nullptr)
      , rvc_session(nullptr)
      , memory_info(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                     OrtMemTypeCPU))

    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::SessionOptions session_options_vec;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        try
        {
            vec_session = std::make_unique<Ort::Session>(
              vec_env, vec_path.c_str(), session_options_vec);
            // std::cout << "Vec model loaded successfully" << std::endl;
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

        // warmup();
    }

    ~OnnxRVC()
    {
        vec_session.reset();
        rvc_session.reset();
    }

    void warmup()
    {
        std::cout << "Warming up the model..." << std::endl;

        // Create dummy data
        int dummy_length = 16000 * 0.5; // 1 second at 16kHz
        std::vector<float> dummy_wav(dummy_length, 0.5f);

        // Forward pass through vec model
        Ort::Value dummy_hubert_tensor = forward_vec_model(dummy_wav);

        // Get the actual shape of the hubert tensor
        auto hubert_shape =
          dummy_hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int dummy_hubert_length = hubert_shape[1];

        // Create dummy pitch data matching the hubert length
        std::vector<float> dummy_pitchf(dummy_hubert_length, 100.0f);
        std::vector<int64_t> dummy_pitch(dummy_hubert_length, 100);
        std::vector<int64_t> dummy_ds = { 1 };

        // Forward pass through RVC model
        try
        {
            forward_rvc_model(dummy_hubert_tensor,
                              dummy_hubert_length,
                              dummy_pitch,
                              dummy_pitchf,
                              dummy_ds);
            std::cout << "Warm-up complete." << std::endl;
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "Error during warm-up: " << e.what() << std::endl;
            // Continue execution even if warm-up fails
        }
    }

    std::vector<float> inference(const std::string& raw_path,
                                 int sid            = 1,
                                 float f0_up_key    = 1,
                                 float pad_time     = 0.5,
                                 float cr_threshold = 0.02)
    {
        // Use AudioFile to load
        AudioFile<float> audioFile;
        bool loaded = audioFile.load(raw_path);
        if (!loaded)
        {
            std::cerr << "Failed to load audio file: " << raw_path << std::endl;
            return std::vector<float>();
        }

        std::vector<float> audio_data = audioFile.samples[0];
        int input_sample_rate         = audioFile.getSampleRate();

        std::cout << "Audio file loaded successfully. Sample rate: "
                  << input_sample_rate << std::endl;

        // Define chunk size and overlap
        const int chunk_size =
          sampling_rate / 2; // Adjust based on your model's requirements
        const int overlap = chunk_size / 16; // 1/8th of chunk size for overlap

        std::vector<float> output_audio;

        for (size_t start = 0; start < audio_data.size();
             start += (chunk_size - overlap))
        {
            size_t end = std::min(start + chunk_size, audio_data.size());
            std::vector<float> chunk(audio_data.begin() + start,
                                     audio_data.begin() + end);

            // Pad the last chunk if necessary
            if (end == audio_data.size())
            {
                chunk.resize(chunk_size, 0.0f);
            }

            // Process the chunk
            auto [pitchf, pitch]     = compute_f0(chunk, f0_up_key);
            Ort::Value hubert_tensor = forward_vec_model(chunk);

            auto hubert_shape =
              hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
            int hubert_length = hubert_shape[1];

            pitchf = resample_vector(pitchf, hubert_length);
            pitch  = resample_vector(pitch, hubert_length);

            std::vector<int64_t> dummy_ds = { 1 };

            std::vector<float> chunk_output = forward_rvc_model(
              hubert_tensor, hubert_length, pitch, pitchf, dummy_ds);

            // Overlap-add
            if (output_audio.empty())
            {
                output_audio = chunk_output;
            }
            else
            {
                // Crossfade the overlapping region
                size_t overlap_samples =
                  overlap * (chunk_output.size() / chunk_size);
                for (size_t i = 0; i < overlap_samples; ++i)
                {
                    float t = static_cast<float>(i) / overlap_samples;
                    output_audio[output_audio.size() - overlap_samples + i] =
                      output_audio[output_audio.size() - overlap_samples + i] *
                        (1 - t) +
                      chunk_output[i] * t;
                }
                // Append the non-overlapping part
                output_audio.insert(output_audio.end(),
                                    chunk_output.begin() + overlap_samples,
                                    chunk_output.end());
            }
        }

        return output_audio;
    }

  private:
    int sampling_rate;
    int hop_size;
    Ort::Env rvc_env;
    Ort::Env vec_env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> vec_session;
    std::unique_ptr<Ort::Session> rvc_session;
    Ort::MemoryInfo memory_info;

    // Add member variables for data flow
    std::vector<float> wav16k;
    Ort::Value hubert_tensor{ nullptr };
    std::vector<float> pitchf;
    std::vector<int64_t> pitch;
    std::vector<int64_t> ds;

    Ort::Value forward_vec_model(const std::vector<float>& wav16k)
    {
        // Prepare input tensor
        std::vector<int64_t> input_shape = {
            1, 1, static_cast<int64_t>(wav16k.size())
        };
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
        auto output_shape =
          output_tensor.GetTensorTypeAndShapeInfo().GetShape();
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

        // Calculate new dimensions
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
                    new_data[b * (time * 2 * channels) + (t * 2) * channels +
                             c]                          = value;
                    new_data[b * (time * 2 * channels) +
                             (t * 2 + 1) * channels + c] = value;
                }
            }
        }

        Ort::Value new_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          new_data.data(),
                                          new_data.size(),
                                          new_shape.data(),
                                          new_shape.size());

        return new_tensor;
    }

    std::pair<std::vector<float>, std::vector<int64_t>> compute_f0(
      const std::vector<float>& wav,
      float up_key)
    {
        return F0Estimator::computeF0WithPitch(wav, up_key);
    }

    template<typename T>
    std::vector<T> resample_vector(const std::vector<T>& input, size_t new_size)
    {
        std::vector<T> resampled(new_size);
        float scale = static_cast<float>(input.size() - 1) / (new_size - 1);
        for (size_t i = 0; i < new_size; ++i)
        {
            float idx        = i * scale;
            size_t idx_floor = static_cast<size_t>(idx);
            size_t idx_ceil  = std::min(idx_floor + 1, input.size() - 1);
            float t          = idx - idx_floor;
            resampled[i] =
              static_cast<T>(input[idx_floor] * (1 - t) + input[idx_ceil] * t);
        }
        return resampled;
    }

    std::vector<float> forward_rvc_model(Ort::Value& hubert_tensor,
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
        const char* input_names[] = {
            "feats", "p_len", "pitch", "pitchf", "sid"
        };
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
        float* output_data =
          output_tensors.front().GetTensorMutableData<float>();
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
            std::cerr << "Warning: NaN values detected in the output."
                      << std::endl;
        }

        // Debug output

        // Padding output
        int padding = 2 * hop_size;
        out_wav.insert(out_wav.end(), padding, 0.0f);

        // Resample from model's sample rate to 16kHz
        std::vector<float> resampled_wav =
          resampleAudio(out_wav, sampling_rate, sr);

        return resampled_wav;
    }

    std::vector<float> pad_audio(const std::vector<float>& audio, int pad_size)
    {
        std::vector<float> padded_audio(audio.size() + pad_size, 0.0f);
        std::copy(audio.begin(), audio.end(), padded_audio.begin());
        return padded_audio;
    }

  public:
    void cleanup()
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
};
