#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "tools/gpu_clock.hpp"

int main(int argc, char **argv) {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    if (props.major < 8) {
        std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
        return 0;
    }

    std::cout << "Using device 0: " << props.name
              << " (SM" << props.major * 10 + props.minor
              << ", " << props.multiProcessorCount
              << ")" << std::endl;

    int m = 5120;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 5120;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    char transA = 'N';
    if (argc >= 5)
        sscanf(argv[4], "%c", &transA);

    char transB = 'T';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transB);

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    for (int j = 0; j < m * k; ++j) {
        h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int j = 0; j < k * n; ++j) {
        h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int j = 0; j < m * n; ++j) {
        h_C[j] = static_cast<TC>(-1);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    } else if (transB == 'T') {
        ldB = n;
    } else {
        assert(false);
    }

    // Run once
    d_C = h_C;

    return 0;
}
