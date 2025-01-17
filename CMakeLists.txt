cmake_minimum_required(VERSION 3.14)
project(rvc_test)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE) 

# Include the ONNX Runtime setup script
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupOnnxRuntime.cmake)

# Find libsamplerate
find_package(PkgConfig REQUIRED)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

# TinyWav
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/../tinywav)

# Option to use portaudio
option(USE_PORTAUDIO "Use PortAudio" OFF)

# Add world pitch detection lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/world)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/AudioFile)




    add_executable(llvc_test 
  
    src/Onnx_RVC.cpp
    src/main.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
    )
    target_include_directories(llvc_test PRIVATE ${BACKEND_BUILD_HEADER_DIRS} ${SAMPLERATE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/vendor/world/src/")
    target_link_directories(llvc_test PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS} ${SAMPLERATE_LIBRARY_DIRS} )
    target_link_libraries(llvc_test PRIVATE onnxruntime world::core ${SAMPLERATE_LIBRARIES} AudioFile)
    # Main executable

endif()




# Copy ONNX Runtime shared library to build directory
if(WIN32)
    add_custom_command(TARGET llvc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:llvc_test>)
endif()

