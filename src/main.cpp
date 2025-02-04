#include "Onnx_RVC.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

int
main()
{
    try
    {
        OnnxRVC rvc("/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
                    "james.onnx",
                    48000,
                    512,
                    "/Users/thomaspower/Developer/Koala/RVC_Test/onnx_models/"
                    "vec-768-layer-12.onnx");

        std::vector<float> output =
          rvc.inference("/Users/thomaspower/Developer/Koala/RVC_Test/"
                        "test_audio/Lead_Vocal_1.wav",
                        1,
                        1.2);

        // Generate a dynamic output file name using the current timestamp
        auto now   = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "/Users/thomaspower/Developer/Koala/RVC_Test/output_audio/output_"
           << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".wav";
        std::string output_file = ss.str();

        myk_tiny::saveWav(output, 1, sr, output_file);
        std::cout << "Output saved to: " << output_file << std::endl;
        rvc.cleanup();
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
