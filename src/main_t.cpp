#include <onnxruntime_cxx_api.h>
#include "../lib/tinywav/myk_tiny.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <samplerate.h>

#include <world/harvest.h>
#include <world/dio.h>
// Forward declaration of utility functions
std::vector<float>
runContentVec(const std::string& modelPath,
              const std::vector<float>& audio,
              int sampleRate);
std::pair<std::vector<float>, std::vector<float>>
estimateF0(const std::vector<double>& wav, int fs, int f0_method = 1);
std::vector<float>
resampleAudio(const std::vector<float>& inputAudio,
              int inputSampleRate,
              int outputSampleRate = 16000);

class OnnxRVC
{
  public:
    OnnxRVC(const std::string model_path,
            int sr                      = 16000,
            int hop_size                = 512,
            const std::string& vec_path = "")
      : sampling_rate(sr)
      , hop_size(hop_size)
      , vec_path(vec_path)
      , model_path(model_path)
    {
    }

    std::vector<float> inference(const std::string& raw_path,
                                 int sid,
                                 const std::string& f0_method = "harvest",
                                 float f0_up_key              = 1,
                                 float pad_time               = 0.5,
                                 float cr_threshold           = 0.02)
    {
        // Load and resample audio
        std::vector<float> wav = myk_tiny::loadWav(raw_path);
        int org_length         = wav.size();

        std::cout << "Original length: " << org_length << std::endl;
        std::vector<float> wav16k = resampleAudio(wav, 16000, 16000);

        // Get hubert features as Ort::Value tensor
        Ort::Value hubert_tensor = forward_vec_model(wav);

        // Get output shape
        auto output_shape =
          hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int64_t hubert_length = output_shape[1]; // Assuming output shape is [1,
                                                 // feats_dynamic_axes_1, 768]

        // Compute F0
        auto [pitchf, pitch] =
          compute_f0(wav, hubert_length, f0_method, f0_up_key);

        // Prepare input tensors
        std::vector<int64_t> ds = { sid };
        std::vector<float> rnd(1 * 192 * hubert_length);
        std::generate(rnd.begin(),
                      rnd.end(),
                      []() { return static_cast<float>(rand()) / RAND_MAX; });

        // Slice and inference
        std::vector<float> output_wav;
        int slice_size = hop_size * 10; // Adjust slice size as needed
        int step_size = hop_size * 5; // Overlap slices for smooth concatenation

        for (int start = 0; start < wav16k.size(); start += step_size)
        {
            int end =
              std::min(start + slice_size, static_cast<int>(wav16k.size()));
            std::vector<float> wav_slice(wav16k.begin() + start,
                                         wav16k.begin() + end);

            // Resample slice to original sample rate if needed
            std::vector<float> slice_output = forward_rvc_model(
              hubert_tensor, hubert_length, pitch, pitchf, ds, rnd);

            // Concatenate slice output
            output_wav.insert(
              output_wav.end(), slice_output.begin(), slice_output.end());
        }

        // Ensure output is correctly sized
        if (output_wav.size() > org_length)
        {
            output_wav.resize(org_length);
        }

        return output_wav;
    }

  private:
    int sampling_rate;
    int hop_size;
    std::string vec_path;
    std::string model_path;

    Ort::Value forward_vec_model(const std::vector<float>& wav16k)
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ContentVecTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session session(env, vec_path.c_str(), session_options);

        // Prepare input tensor
        std::vector<int64_t> input_shape = {
            1, 1, static_cast<int64_t>(wav16k.size())
        };
        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          const_cast<float*>(wav16k.data()),
                                          wav16k.size(),
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

        // Return the first output tensor
        return std::move(output_tensors.front());
    }

    std::pair<std::vector<float>, std::vector<int64_t>> compute_f0(
      const std::vector<float>& wav,
      int64_t length,
      const std::string& method,
      float up_key)
    {
        // Convert wav to double precision for Harvest
        std::vector<double> wav_double(wav.begin(), wav.end());

        // Determine f0_method
        int f0_method =
          (method == "harvest") ? 0 : 1; // Assuming 0 for Harvest, 1 for Dio

        // Estimate F0
        auto [f0, temporal_positions] = estimateF0(wav_double, 16000, 1);

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

        // Resize f0 and pitch to match the required length
        if (f0.size() != length)
        {
            f0    = resample_vector(f0, length);
            pitch = resample_vector(pitch, length);
        }

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
                                         int64_t hubert_length,
                                         const std::vector<int64_t>& pitch,
                                         const std::vector<float>& pitchf,
                                         const std::vector<int64_t>& ds,
                                         const std::vector<float>& rnd)
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RVCModelTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session model(env, model_path.c_str(), session_options);
        // Prepare input tensors
        std::vector<int64_t> hubert_shape = { 1, hubert_length, 768 };
        std::vector<int64_t> length_shape = { 1 };

        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info,
          const_cast<int64_t*>(&hubert_length),
          1,
          length_shape.data(),
          length_shape.size());
        Ort::Value pitch_tensor =
          Ort::Value::CreateTensor<int64_t>(memory_info,
                                            const_cast<int64_t*>(pitch.data()),
                                            pitch.size(),
                                            hubert_shape.data(),
                                            hubert_shape.size());
        Ort::Value pitchf_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          const_cast<float*>(pitchf.data()),
                                          pitchf.size(),
                                          hubert_shape.data(),
                                          hubert_shape.size());
        Ort::Value ds_tensor =
          Ort::Value::CreateTensor<int64_t>(memory_info,
                                            const_cast<int64_t*>(ds.data()),
                                            ds.size(),
                                            length_shape.data(),
                                            length_shape.size());
        Ort::Value rnd_tensor =
          Ort::Value::CreateTensor<float>(memory_info,
                                          const_cast<float*>(rnd.data()),
                                          rnd.size(),
                                          hubert_shape.data(),
                                          hubert_shape.size());

        const char* input_names[]  = { "source", "length", "pitch",
                                       "pitchf", "ds",     "rnd" };
        const char* output_names[] = { "audio" };

        // Run inference
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(6); // Reserve space for 6 tensors

        input_tensors.push_back(std::move(hubert_tensor));
        input_tensors.push_back(std::move(length_tensor));
        input_tensors.push_back(std::move(pitch_tensor));
        input_tensors.push_back(std::move(pitchf_tensor));
        input_tensors.push_back(std::move(ds_tensor));
        // Run inference
        auto output_tensors = model.Run(Ort::RunOptions{ nullptr },
                                        input_names,
                                        input_tensors.data(),
                                        input_tensors.size(),
                                        output_names,
                                        1);

        // Retrieve output tensor
        auto output_tensor = std::move(output_tensors.front());
        float* output_data = output_tensor.GetTensorMutableData<float>();
        std::vector<float> output(
          output_data,
          output_data +
            output_tensor.GetTensorTypeAndShapeInfo().GetShape().back());

        return output;
    }
};

// Utility functions

std::vector<float>
resampleAudio(const std::vector<float>& inputAudio,
              int inputSampleRate,
              int outputSampleRate)
{
    double ratio      = static_cast<double>(outputSampleRate) / inputSampleRate;
    size_t outputSize = static_cast<size_t>(inputAudio.size() * ratio);
    std::vector<float> outputAudio(outputSize);

    SRC_DATA srcData;
    srcData.data_in       = inputAudio.data();
    srcData.input_frames  = inputAudio.size();
    srcData.data_out      = outputAudio.data();
    srcData.output_frames = outputAudio.size();
    srcData.src_ratio     = ratio;
    srcData.end_of_input  = 0;

    int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
    if (error)
    {
        throw std::runtime_error("Error during resampling: " +
                                 std::string(src_strerror(error)));
    }

    outputAudio.resize(srcData.output_frames_gen);
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

int
main()
{
    try
    {
        OnnxRVC rvcModel("/Users/thomaspower/Developer/Koala/RVC_Test/"
                         "onnx_models/amitaro_v2_16k.onnx",
                         16000,
                         512,
                         "/Users/thomaspower/Developer/Koala/RVC_Test/"
                         "onnx_models/vec-768-layer-12.onnx");
        std::vector<float> output =
          rvcModel.inference("/Users/thomaspower/Developer/Koala/RVC_Test/"
                             "test_audio/Lead_Vocal_16.wav",
                             1,
                             "harvest",
                             1.0,
                             0.5,
                             0.02);
        myk_tiny::saveWav(output,
                          1,
                          16000,
                          "/Users/thomaspower/Developer/Koala/RVC_Test/"
                          "output_audio/output.wav");
        std::cout << "Inference completed successfully!" << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        std::cerr << "Error code: " << e.GetOrtErrorCode() << std::endl;

        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
