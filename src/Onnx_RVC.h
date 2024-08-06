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

static const int sr = 48000;

std::pair<std::vector<float>, std::vector<float>>
estimateF0(const std::vector<double>& wav, int fs, int f0_method = 1)
{
    HarvestOption option;
    InitializeHarvestOption(&option);

    // You can adjust these parameters if needed
    option.f0_floor     = 50.0;   // Hz
    option.f0_ceil      = 1100.0; // Hz
    option.frame_period = 10.0;   // ms

    if (f0_method == 0)
    { // Harvest

        int f0_length =
          GetSamplesForHarvest(fs, wav.size(), option.frame_period);
        std::vector<double> f0(f0_length);
        std::vector<double> temporal_positions(f0_length);
        Harvest(wav.data(),
                wav.size(),
                fs,
                &option,
                temporal_positions.data(),
                f0.data());
        // Recast the f0 and temporal_positions vectors to float
        std::vector<float> f0_float(f0.begin(), f0.end());
        std::vector<float> temporal_positions_float(temporal_positions.begin(),
                                                    temporal_positions.end());
        return { f0_float, temporal_positions_float };
    }
    else
    {

        DioOption dio_option;
        InitializeDioOption(&dio_option);
        dio_option.f0_floor           = option.f0_floor;
        dio_option.f0_ceil            = option.f0_ceil;
        dio_option.channels_in_octave = 2.0;
        dio_option.frame_period       = option.frame_period;
        dio_option.speed              = 1;
        dio_option.allowed_range      = 0.1;

        int f0_length =
          GetSamplesForDIO(fs, wav.size(), dio_option.frame_period);
        std::vector<double> f0(f0_length);
        std::vector<double> temporal_positions(f0_length);

        Dio(wav.data(),
            wav.size(),
            fs,
            &dio_option,
            temporal_positions.data(),
            f0.data());

        // Recast the f0 and temporal_positions vectors to float
        std::vector<float> f0_float(f0.begin(), f0.end());
        std::vector<float> temporal_positions_float(temporal_positions.begin(),
                                                    temporal_positions.end());

        return { f0_float, temporal_positions_float };
    }
}

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
        session_options.SetIntraOpNumThreads(0);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::SessionOptions session_options_vec;
        session_options.SetIntraOpNumThreads(6);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        // Set parallel
        session_options.SetExecutionMode(ORT_PARALLEL);

        try
        {
            vec_session = std::make_unique<Ort::Session>(
              vec_env, vec_path.c_str(), session_options_vec);
            std::cout << "Vec model loaded successfully" << std::endl;
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
        std::cout << "Audio file loaded successfully"
                  << audioFile.getSampleRate() << std::endl;

        auto [pitchf, pitch] = compute_f0(audio_data, f0_up_key);

        // std::vector<float> downsampled =
        //   resampleAudio(audio_data, audioFile.getSampleRate(), 8000);

        // Forward pass through vec model
        Ort::Value dummy_hubert_tensor = forward_vec_model(audio_data);

        // Get the actual shape of the hubert tensor
        auto hubert_shape =
          dummy_hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int dummy_hubert_length = hubert_shape[1];

        pitchf = resample_vector(pitchf, dummy_hubert_length);
        pitch  = resample_vector(pitch, dummy_hubert_length);

        // // Create dummy pitch data matching the hubert length
        // std::vector<float> pitchf(dummy_hubert_length, 100.0f);
        // std::vector<int64_t> pitch(dummy_hubert_length, 50);
        std::vector<int64_t> dummy_ds = { 1 };

        // Forward pass through RVC model
        try
        {
            audio_data = forward_rvc_model(dummy_hubert_tensor,
                                           dummy_hubert_length,
                                           pitch,
                                           pitchf,
                                           dummy_ds);
            std::cout << "Warm-up complete." << std::endl;
            return audio_data;
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "Error during warm-up: " << e.what() << std::endl;
            return std::vector<float>();
            // Continue execution even if warm-up fails
        }
        // auto start_time = std::chrono::high_resolution_clock::now();

        // // Load and resample audio
        // std::vector<float> wav    = myk_tiny::loadWav(raw_path);
        // int org_length            = wav.size();
        // std::vector<float> wav16k = resampleAudio(wav, 16000, 16000);

        // std::cout << "Loaded audio length: " << wav16k.size() << std::endl;

        // auto hubert_start = std::chrono::high_resolution_clock::now();
        // // Forward pass through vec model
        // Ort::Value hubert_tensor = forward_vec_model(wav16k);
        // auto hubert_end          = std::chrono::high_resolution_clock::now();

        // // Get the actual shape of the hubert tensor
        // auto hubert_shape =
        //   hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
        // int hubert_length = hubert_shape[1];

        // std::cout << "Hubert tensor shape: ";
        // for (const auto& dim : hubert_shape)
        // {
        //     std::cout << dim << " ";
        // }
        // std::cout << std::endl;

        // // Create pitch data matching the hubert length
        // auto [pitchf, pitch]    = compute_f0(wav16k, hubert_length,
        // f0_up_key); std::vector<int64_t> ds = { 1 };

        // auto rvc_start = std::chrono::high_resolution_clock::now();
        // // Forward pass through RVC model
        // auto out_wav =
        //   forward_rvc_model(hubert_tensor, hubert_length, pitch, pitchf, ds);
        // auto rvc_end = std::chrono::high_resolution_clock::now();

        // auto end_time = std::chrono::high_resolution_clock::now();

        // // Calculate durations
        // auto total_duration =
        //   std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
        //                                                         start_time)
        //     .count();
        // auto hubert_duration =
        //   std::chrono::duration_cast<std::chrono::milliseconds>(hubert_end -
        //                                                         hubert_start)
        //     .count();
        // auto rvc_duration =
        //   std::chrono::duration_cast<std::chrono::milliseconds>(rvc_end -
        //                                                         rvc_start)
        //     .count();

        // // Calculate audio duration
        // float audio_duration = static_cast<float>(org_length) / 16000;

        // std::cout << "Total processing time: " << total_duration << " ms"
        //           << std::endl;
        // std::cout << "Hubert processing time: " << hubert_duration << " ms"
        //           << std::endl;
        // std::cout << "RVC model inference time: " << rvc_duration << " ms"
        //           << std::endl;
        // std::cout << "Audio duration: " << audio_duration * 1000 << " ms"
        //           << std::endl;
        // std::cout << "Real-time factor: "
        //           << (total_duration / (audio_duration * 1000)) << std::endl;

        // return out_wav;
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
        // Convert wav to double precision for Harvest
        std::vector<double> wav_double(wav.begin(), wav.end());

        // Determine f0_method
        int f0_method = 1;

        // Estimate F0
        auto [f0, temporal_positions] =
          estimateF0(wav_double, 16000, f0_method);

        // Apply up_key
        for (auto& f : f0)
        {
            f *= up_key;
        }

        // Compute f0_mel
        float f0_min     = 50.0f;
        float f0_max     = 1100.0f;
        float f0_mel_min = 1127.0f * std::log(1.0f + f0_min / 700.0f);
        float f0_mel_max = 1127.0f * std::log(1.0f + f0_max / 700.0f);

        std::vector<int64_t> pitch(f0.size());
        for (size_t i = 0; i < f0.size(); ++i)
        {
            float f0_mel = 1127.0f * std::log(1.0f + f0[i] / 700.0f);
            if (f0_mel > 0.0f)
            {
                f0_mel =
                  (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) +
                  1.0f;
            }
            f0_mel   = std::clamp(f0_mel, 1.0f, 255.0f);
            pitch[i] = static_cast<int64_t>(std::round(f0_mel));
        }

        // std::cout << "F0 size: " << f0.size() << std::endl;
        // std::cout << "Pitch size: " << pitch.size() << std::endl;

        return { f0, pitch };
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
        std::cout << "Hubert tensor shape: ";
        for (const auto& dim :
             hubert_tensor.GetTensorTypeAndShapeInfo().GetShape())
        {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Hubert length: " << hubert_length << std::endl;
        std::cout << "Pitch size: " << pitch.size() << std::endl;
        std::cout << "Pitchf size: " << pitchf.size() << std::endl;
        std::cout << "DS size: " << ds.size() << std::endl;

        float* hubert_data = hubert_tensor.GetTensorMutableData<float>();
        std::cout << "First 15 values of Hubert tensor: ";
        for (int i = 0; i < std::min(15, hubert_length * 768); ++i)
        {
            std::cout << hubert_data[i] << " ";

            if (std::isnan(hubert_data[i]))
            {
                return std::vector<float>();
            }
        }
        std::cout << std::endl;

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
        std::cout << "Output size: " << out_wav.size() << std::endl;
        if (!out_wav.empty())
        {
            std::cout << "First 15 values: ";
            for (int i = 0; i < std::min(15, static_cast<int>(out_wav.size()));
                 ++i)
            {
                std::cout << out_wav[i] << " ";
            }
            std::cout << std::endl;
        }

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
