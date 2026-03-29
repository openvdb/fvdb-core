// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_TESTS_TEST_UTILS_CUH
#define DISPATCH_TESTS_TEST_UTILS_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace dispatch {
namespace test {

// =============================================================================
// RAII device buffer
// =============================================================================

template <typename T> class DeviceBuffer {
  public:
    explicit DeviceBuffer(int64_t n) : _size(n) {
        if (n > 0) {
            auto err = cudaMalloc(&_data, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
        } else {
            _data = nullptr;
        }
    }

    ~DeviceBuffer() {
        if (_data != nullptr) {
            cudaFree(_data);
        }
    }

    // Non-copyable
    DeviceBuffer(DeviceBuffer const &)            = delete;
    DeviceBuffer &operator=(DeviceBuffer const &) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer &&other) noexcept : _data(other._data), _size(other._size) {
        other._data = nullptr;
        other._size = 0;
    }

    DeviceBuffer &
    operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            if (_data != nullptr) {
                cudaFree(_data);
            }
            _data       = other._data;
            _size       = other._size;
            other._data = nullptr;
            other._size = 0;
        }
        return *this;
    }

    T *
    get() const {
        return _data;
    }
    int64_t
    size() const {
        return _size;
    }

  private:
    T *_data = nullptr;
    int64_t _size;
};

// =============================================================================
// Host-device copy helpers
// =============================================================================

template <typename T>
void
copy_to_device(T const *host, T *device, int64_t n) {
    if (n > 0) {
        auto err = cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy HtoD failed");
        }
    }
}

template <typename T>
void
copy_to_host(T const *device, T *host, int64_t n) {
    if (n > 0) {
        auto err = cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy DtoH failed");
        }
    }
}

} // namespace test
} // namespace dispatch

#endif // DISPATCH_TESTS_TEST_UTILS_CUH
