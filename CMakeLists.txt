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




if(USE_PORTAUDIO)
    add_executable(llvc_test_pa 
    src/main_pa_threading.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
    )
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    include_directories(${PORTAUDIO_INCLUDE_DIRS})
    target_include_directories(llvc_test_pa PRIVATE ${BACKEND_BUILD_HEADER_DIRS} ${PORTAUDIO_INCLUDE_DIRS} ${SAMPLERATE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/vendor/world/src/")
    target_link_directories(llvc_test_pa PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS} ${PORTAUDIO_LIBRARY_DIRS} )
    target_link_libraries(llvc_test_pa onnxruntime world::core ${PORTAUDIO_LIBRARIES} ${SAMPLERATE_LIBRARIES})
    message(STATUS "Using PortAudio")
    # Main executable

else()
    add_executable(llvc_test 
    src/main.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
    )
    target_include_directories(llvc_test PRIVATE ${BACKEND_BUILD_HEADER_DIRS} ${SAMPLERATE_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/vendor/world/src/")
    target_link_directories(llvc_test PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS} ${SAMPLERATE_LIBRARY_DIRS} )
    target_link_libraries(llvc_test PRIVATE onnxruntime world::core ${SAMPLERATE_LIBRARIES})
    # Main executable

endif()

endif()


# Copy ONNX Runtime shared library to build directory
if(WIN32)
    add_custom_command(TARGET llvc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:llvc_test>)
endif()

