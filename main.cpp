#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
extern void sq_distance_batch_cuda(const float* a, const float* b, float* out, int n, int m, cudaStream_t stream = 0);
#endif

extern void sq_distance_batch_cpu(const float* a, const float* b, float* out, int n, int m);

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

int main(int argc, char** argv) {
    const int n = 1024;
    const int m = 10000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> a(n);
    std::vector<float> b(n * m);
    std::vector<float> out_cpu(m);

    for (int i = 0; i < n; ++i) a[i] = dist(rng);
    for (int i = 0; i < n * m; ++i) b[i] = dist(rng);

    auto start = std::chrono::high_resolution_clock::now();
    sq_distance_batch_cpu(a.data(), b.data(), out_cpu.data(), n, m);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU: " << cpu_ms << " ms" << std::endl;

#if defined(USE_CUDA)
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Warning: No CUDA-capable GPU found, skipping CUDA benchmark" << std::endl;
    } else {
        std::vector<float> out_cuda(m);
        float *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * m * sizeof(float));
        cudaMalloc(&d_out, m * sizeof(float));

        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaEventRecord(start_event);
        sq_distance_batch_cuda(d_a, d_b, d_out, n, m);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);

        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start_event, stop_event);
        std::cout << "CUDA: " << cuda_ms << " ms" << std::endl;

        cudaMemcpy(out_cuda.data(), d_out, m * sizeof(float), cudaMemcpyDeviceToHost);

        float diff = max_abs_diff(out_cpu.data(), out_cuda.data(), m);
        std::cout << "Max abs diff: " << diff << std::endl;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
#else
    std::cerr << "Warning: CUDA not available at compile time" << std::endl;
#endif

    return 0;
}