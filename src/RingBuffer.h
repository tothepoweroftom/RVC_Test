#pragma once
#include <vector>
#include <atomic>

class RingBuffer
{
  public:
    RingBuffer(size_t size)
      : buffer_(size)
      , writePos_(0)
      , readPos_(0)
      , size_(size)
    {
    }

    void write(const float* data, size_t numSamples)
    {
        size_t written = 0;
        while (written < numSamples)
        {
            size_t available = size_ - writePos_;
            size_t remaining = numSamples - written;
            size_t toWrite   = std::min(available, remaining);

            std::copy(data + written,
                      data + written + toWrite,
                      buffer_.begin() + writePos_);

            written += toWrite;
            writePos_ = (writePos_ + toWrite) % size_;
        }
    }

    void read(float* data, size_t numSamples)
    {
        size_t read = 0;
        while (read < numSamples)
        {
            size_t available = getAvailableRead();
            size_t remaining = numSamples - read;
            size_t toRead    = std::min(available, remaining);

            std::copy(buffer_.begin() + readPos_,
                      buffer_.begin() + readPos_ + toRead,
                      data + read);

            read += toRead;
            readPos_ = (readPos_ + toRead) % size_;
        }
    }

    size_t getAvailableRead() const
    {
        return (writePos_ - readPos_ + size_) % size_;
    }

  private:
    std::vector<float> buffer_;
    std::atomic<size_t> writePos_;
    std::atomic<size_t> readPos_;
    const size_t size_;
};