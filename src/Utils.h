#pragma once

#include <samplerate.h>

namespace Utils
{

// TODO: Replace with r8brain
// Downsampling function
static std::vector<float>
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

// Resample a vector to a new size
template<typename T>
static std::vector<T>
resample_vector(const std::vector<T>& input, size_t new_size)
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

// Pad the audio with zeros
static std::vector<float>
pad_audio(const std::vector<float>& audio, int pad_size)
{
    std::vector<float> padded_audio(audio.size() + pad_size, 0.0f);
    std::copy(audio.begin(), audio.end(), padded_audio.begin());
    return padded_audio;
}
} // namespace Utils