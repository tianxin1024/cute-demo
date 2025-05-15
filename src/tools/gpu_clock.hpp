#pragma once

#include <cuda_runtime.h>

struct GPU_Clock {
    GPU_Clock() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }

    ~GPU_Clock() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    float milliseconds() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time;
    }

    float seconds() {
        return milliseconds() * float(1e-3);
    }

private:
    cudaEvent_t start_, stop_;
};
