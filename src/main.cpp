#include "Onnx_RVC.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <AudioFile.h>

int
main()
{
    try
    {
        int output_sample_rate = 40000;
        OnnxRVC rvc("/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
                    "jacob_paul_3_simple.onnx",
                    output_sample_rate,
                    "/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
                    "vec-768-layer-12.onnx");

        // Path to test audio
        std::string raw_path =
          "/Users/thomaspower/Developer/Koala/RVC_Test/test_audio/tom2.wav";

        // Load test audio
        // Use AudioFile to load
        AudioFile<float> audioFile;
        bool loaded = audioFile.load(raw_path);
        if (!loaded)
        {
            std::cerr << "Failed to load audio file: " << raw_path << std::endl;
            return 1;
        }
        // Mono audio
        std::vector<float> audio_data = audioFile.samples[0];
        int input_sample_rate         = audioFile.getSampleRate();

        std::cout << "Audio file loaded successfully. Sample rate: "
                  << input_sample_rate << std::endl;

        // // Call inference
        // std::vector<float> output =
        //   rvc.inference(audio_data, input_sample_rate);

        // Define chunk size and overlap
        const int chunk_size = 16000;          // 0.5 second chunks
        const int overlap    = chunk_size / 2; // 1/16th overlap

        std::vector<float> final_output;

        // Process audio in chunks
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
            std::vector<float> chunk_output =
              rvc.inference(chunk, input_sample_rate);

            // Overlap-add
            if (final_output.empty())
            {
                final_output = chunk_output;
            }
            else
            {
                // Crossfade the overlapping region
                size_t overlap_samples =
                  overlap * (chunk_output.size() / chunk_size);
                for (size_t i = 0; i < overlap_samples; ++i)
                {
                    float t = static_cast<float>(i) / overlap_samples;
                    final_output[final_output.size() - overlap_samples + i] =
                      final_output[final_output.size() - overlap_samples + i] *
                        (1 - t) +
                      chunk_output[i] * t;
                }
                // Append the non-overlapping part
                final_output.insert(final_output.end(),
                                    chunk_output.begin() + overlap_samples,
                                    chunk_output.end());
            }
        }

        std::vector<float> output = final_output;

        // Generate a dynamic output file name using the current timestamp
        auto now   = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "/Users/thomaspower/Developer/Koala/RVC_Test/output_audio/output_"
           << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".wav";
        std::string output_file = ss.str();

        // Save output with AudioFile
        AudioFile<float> outputFile;
        outputFile.setSampleRate(output_sample_rate);
        outputFile.setNumChannels(1);
        outputFile.samples[0] = output;
        outputFile.save(output_file);
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
