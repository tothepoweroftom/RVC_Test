#include <onnxruntime_cxx_api.h>
#include "../lib/tinywav/myk_tiny.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <samplerate.h>
#include <sstream>

#include <world/harvest.h>
#include <world/dio.h>

static const int sr = 48000;

std::vector<float>
runContentVec(const std::string& modelPath,
              const std::vector<float>& audio,
              int sampleRate)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ContentVecTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(12);
    session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);

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

std::pair<std::vector<float>, std::vector<float>>
estimateF0(const std::vector<double>& wav, int fs, int f0_method = 0)
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
    OnnxRVC(const std::string model_path =
              "/Users/thomaspower/Developer/Koala/"
              "RVC_Test/onnx_models/jojo-model.onnx",
            int sr       = 16000,
            int hop_size = 512,
            const std::string& vec_path =
              "/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
              "vec-768-layer-12.onnx")
      : sampling_rate(sr)
      , hop_size(hop_size)
      , vec_path(vec_path)
      , model_path(model_path)

    {
        ////std::cout << "Sampling rate: " << sampling_rate << std::endl;
    }

    std::vector<float> inference(const std::string& raw_path,
                                 int sid,
                                 float f0_up_key    = 1,
                                 float pad_time     = 0.5,
                                 float cr_threshold = 0.02)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Load and resample audio
        std::vector<float> wav    = myk_tiny::loadWav(raw_path);
        int org_length            = wav.size();
        std::vector<float> wav16k = resampleAudio(wav, 16000, 16000);

        auto hubert_start = std::chrono::high_resolution_clock::now();
        // Get hubert features as Ort::Value tensor
        Ort::Value hubert_tensor = forward_vec_model(wav16k);
        auto hubert_end          = std::chrono::high_resolution_clock::now();

        // Get output shape
        auto output_shape =
          hubert_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int hubert_length = output_shape[1];

        auto f0_start = std::chrono::high_resolution_clock::now();
        // Compute F0
        auto [pitchf, pitch] = compute_f0(wav, hubert_length, f0_up_key);
        auto f0_end          = std::chrono::high_resolution_clock::now();

        // Prepare input tensors
        std::vector<int64_t> ds = { sid };

        auto rvc_start = std::chrono::high_resolution_clock::now();
        // Forward pass through RVC model
        auto out_wav =
          forward_rvc_model(hubert_tensor, hubert_length, pitch, pitchf, ds);
        auto rvc_end = std::chrono::high_resolution_clock::now();

        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate durations
        auto total_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                start_time)
            .count();
        auto hubert_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(hubert_end -
                                                                hubert_start)
            .count();
        auto f0_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(f0_end -
                                                                f0_start)
            .count();
        auto rvc_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(rvc_end -
                                                                rvc_start)
            .count();

        // Calculate audio duration
        float audio_duration = static_cast<float>(org_length) / 16000;

        std::cout << "Total processing time: " << total_duration << " ms"
                  << std::endl;
        std::cout << "Hubert processing time: " << hubert_duration << " ms"
                  << std::endl;
        std::cout << "F0 computation time: " << f0_duration << " ms"
                  << std::endl;
        std::cout << "RVC model inference time: " << rvc_duration << " ms"
                  << std::endl;
        std::cout << "Audio duration: " << audio_duration * 1000 << " ms"
                  << std::endl;
        std::cout << "Real-time factor: "
                  << (total_duration / (audio_duration * 1000)) << std::endl;

        return out_wav;
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
        session_options.SetIntraOpNumThreads(12);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_BASIC);

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

        // Get output tensor
        Ort::Value& output_tensor = output_tensors.front();

        // Get output shape and data
        auto output_shape =
          output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        float* output_data = output_tensor.GetTensorMutableData<float>();

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

    std::pair<std::vector<float>, std::vector<int64_t>>
    compute_f0(const std::vector<float>& wav, int length, float up_key)
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

        // Resize f0 and pitch to match the required length
        if (f0.size() != length)
        {
            f0    = resample_vector(f0, length);
            pitch = resample_vector(pitch, length);
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
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RVCModelTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(12);
        session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_BASIC);

        Ort::Session model(env, model_path.c_str(), session_options);
        // Prepare input tensors
        std::vector<int64_t> hubert_shape = { 1, hubert_length, 768 };
        std::vector<int64_t> length_shape = { 1 };
        std::vector<int64_t> pitch_shape  = { 1, hubert_length };
        std::vector<int64_t> ds_shape     = { 1 };

        Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

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
        // Define input and output names
        const char* input_names[] = {
            "feats", "p_len", "pitch", "pitchf", "sid"
        };
        const char* output_names[] = { "audio" };

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

        // Print output tensor shape
        auto output_shape =
          output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // std::cout << "Output tensor shape: ";
        for (const auto& dim : output_shape)
        {
            // std::cout << dim << " ";
        }
        // std::cout << std::endl;

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
};

int
main()
{
    try
    {
        OnnxRVC rvc("/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
                    "trentreznor-48k_simple.onnx",
                    sr);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> output =
          rvc.inference("/Users/thomaspower/Developer/Koala/RVC_Test/"
                        "test_audio/Lead_Vocal_16.wav",
                        0,
                        0.45);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();

        std::cout << "Total execution time: " << duration << " ms" << std::endl;

        // Generate a dynamic output file name using the current timestamp
        auto now   = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss
          << "/Users/thomaspower/Developer/Koala/RVC_Test/output_audio2/output_"
          << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".wav";
        std::string output_file = ss.str();

        myk_tiny::saveWav(output, 1, sr, output_file);
        std::cout << "Output saved to: " << output_file << std::endl;
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
