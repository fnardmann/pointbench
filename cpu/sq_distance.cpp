#include <cmath>

float sq_distance_cpu(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void sq_distance_batch_cpu(const float* a, const float* b, float* out, int n, int m) {
    for (int j = 0; j < m; ++j) {
        out[j] = sq_distance_cpu(a, b + j * n, n);
    }
}