#pragma once
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>
#include <numeric>
#include <AudioFile.h>

#include "F0Estimator.h"
#include "Utils.h"

static const int sr = 40000;

class OnnxRVC
{
  public:
    OnnxRVC(const std::string& model_path,
            int model_sample_rate       = 16000,
            const std::string& vec_path = "/path/to/vec-768-layer-12.onnx");

    ~OnnxRVC();

    std::vector<float> inference(std::vector<float> audio_data,
                                 float input_sample_rate);

    void cleanup();

  private:
    int model_sample_rate;
    Ort::Env env;
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

    RingBuffer inputBuffer_;
    RingBuffer outputBuffer_;

    Ort::Value forward_vec_model(const std::vector<float>& audio,
                                 int samplerate);

    std::pair<std::vector<float>, std::vector<int64_t>> compute_f0(
      const std::vector<float>& wav,
      float up_key);

    std::vector<float> forward_rvc_model(Ort::Value& hubert_tensor,
                                         int hubert_length,
                                         const std::vector<int64_t>& pitch,
                                         const std::vector<float>& pitchf,
                                         const std::vector<int64_t>& ds);
};
