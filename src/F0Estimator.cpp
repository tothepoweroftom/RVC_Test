#include "F0Estimator.h"

F0Estimator::F0Result
F0Estimator::estimate(const std::vector<double>& wav,
                      int sample_rate,
                      int f0_method,
                      const F0Config& config)
{
    if (f0_method == 0)
    {
        return estimateHarvest(wav, sample_rate, config);
    }
    else
    {
        return estimateDio(wav, sample_rate, config);
    }
}

F0Estimator::F0Result
F0Estimator::estimateHarvest(const std::vector<double>& wav,
                             int sample_rate,
                             const F0Config& config)
{
    HarvestOption option;
    InitializeHarvestOption(&option);

    option.f0_floor     = config.f0_floor;
    option.f0_ceil      = config.f0_ceil;
    option.frame_period = config.frame_period;

    int f0_length =
      GetSamplesForHarvest(sample_rate, wav.size(), option.frame_period);

    F0Result result;
    result.f0.resize(f0_length);
    result.temporal_positions.resize(f0_length);

    Harvest(wav.data(),
            wav.size(),
            sample_rate,
            &option,
            result.temporal_positions.data(),
            result.f0.data());

    return result;
}

F0Estimator::F0Result
F0Estimator::estimateDio(const std::vector<double>& wav,
                         int sample_rate,
                         const F0Config& config)
{
    DioOption option;
    InitializeDioOption(&option);

    option.f0_floor           = config.f0_floor;
    option.f0_ceil            = config.f0_ceil;
    option.channels_in_octave = config.channels_in_octave;
    option.frame_period       = config.frame_period;
    option.speed              = config.speed;
    option.allowed_range      = config.allowed_range;

    int f0_length =
      GetSamplesForDIO(sample_rate, wav.size(), option.frame_period);

    F0Result result;
    result.f0.resize(f0_length);
    result.temporal_positions.resize(f0_length);

    Dio(wav.data(),
        wav.size(),
        sample_rate,
        &option,
        result.temporal_positions.data(),
        result.f0.data());

    return result;
}

std::pair<std::vector<float>, std::vector<int64_t>>
F0Estimator::computeF0WithPitch(const std::vector<float>& wav,
                                float up_key,
                                int sample_rate)
{
    // Convert wav to double precision for Harvest
    std::vector<double> wav_double(wav.begin(), wav.end());

    // Estimate F0
    auto result = estimate(wav_double, sample_rate, 1);

    // Convert double f0 to float for return value
    std::vector<float> f0_float(result.f0.begin(), result.f0.end());

    // Apply up_key
    for (auto& f : f0_float)
    {
        f *= up_key;
    }

    // Convert F0 to pitch
    std::vector<int64_t> pitch = convertF0ToPitch(f0_float);

    return { f0_float, pitch };
}

std::vector<int64_t>
F0Estimator::convertF0ToPitch(const std::vector<float>& f0)
{
    const float f0_min     = 50.0f;
    const float f0_max     = 1100.0f;
    const float f0_mel_min = 1127.0f * std::log(1.0f + f0_min / 700.0f);
    const float f0_mel_max = 1127.0f * std::log(1.0f + f0_max / 700.0f);

    std::vector<int64_t> pitch(f0.size());
    for (size_t i = 0; i < f0.size(); ++i)
    {
        float f0_mel = 1127.0f * std::log(1.0f + f0[i] / 700.0f);
        if (f0_mel > 0.0f)
        {
            f0_mel =
              (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) + 1.0f;
        }
        f0_mel   = std::clamp(f0_mel, 1.0f, 255.0f);
        pitch[i] = static_cast<int64_t>(std::round(f0_mel));
    }

    return pitch;
}