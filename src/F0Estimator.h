#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <world/harvest.h>
#include <world/dio.h>

class F0Estimator
{
  public:
    struct F0Config
    {
        float f0_floor;
        float f0_ceil;
        float frame_period;
        float channels_in_octave;
        int speed;
        float allowed_range;

        // Constructor with default values
        F0Config()
          : f0_floor(50.0f)
          , f0_ceil(1100.0f)
          , frame_period(10.0f)
          , channels_in_octave(2.0f)
          , speed(1)
          , allowed_range(0.1f)
        {
        }
    };

    struct F0Result
    {
        std::vector<double> f0;
        std::vector<double> temporal_positions;
    };

    static F0Result estimate(const std::vector<double>& wav,
                             int sample_rate,
                             int f0_method          = 1,
                             const F0Config& config = F0Config());

    static std::pair<std::vector<float>, std::vector<int64_t>>
    computeF0WithPitch(const std::vector<float>& wav,
                       float up_key,
                       int sample_rate = 16000);

  private:
    static F0Result estimateHarvest(const std::vector<double>& wav,
                                    int sample_rate,
                                    const F0Config& config);

    static F0Result estimateDio(const std::vector<double>& wav,
                                int sample_rate,
                                const F0Config& config);

    static std::vector<int64_t> convertF0ToPitch(const std::vector<float>& f0);
};