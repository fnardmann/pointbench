#include <cuda_runtime.h>

__device__ float sq_distance_device(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

__global__ void sq_distance_batch_kernel(const float* a, const float* b, float* out, int n, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) {
        out[j] = sq_distance_device(a, b + j * n, n);
    }
}

void sq_distance_batch_cuda(const float* a, const float* b, float* out, int n, int m, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    sq_distance_batch_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n, m);
}