#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define  OPS_SCALE   (2048)

#define KERNEL_OPS_COUNT    (2 * OPS_SCALE)

__global__ void kernel_gflops(float* a, float* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= OPS_SCALE * OPS_SCALE) {
        return;
    }
    a[idx] = b[idx]; // 1 LOAD + 1 STORE, 0 FP32 ops, 0 FP64 ops

    /**
     * ~TODO~
     * Measure FP32 GFlops and FP64 Gflops of the device
     * Try and achieve close to theoretical peak performance
     */

    float x = a[idx]; // 1 LOAD, 0 FP32 ops, 0 FP64 ops

    //Make ops_scale number of operations
    //FP32: 2 ops per iteration
    for (int i = 0; i < OPS_SCALE; i++) {
        x = x * x + x; // 2 FP32 ops, 0 FP64 ops
    }
    b[idx] = x;
    // double y = (double) x;
    // FP64: 2 op per iteration
    // for (int i = 0; i < OPS_SCALE; i++) {
    //     y = y * y + y; // 2 FP64 ops, 0 FP32 ops
    // }
}

void fill_array_int(int* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }
}

void fill_array_float(float* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
    }
}

void fill_array_random(float* a, int N) {
    for (int i = 0; i < N; ++i) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    float* device_a = 0;
    float* device_b = 0;
    float* host_a = 0;
    float* host_b = 0;

    int size = OPS_SCALE * OPS_SCALE;

    // Arrays a and b are of size N * N
    host_a = (float*)malloc(size * sizeof(float));
    host_b = (float*)malloc(size * sizeof(float));
    cudaMalloc((void**)&device_a, size * sizeof(float));
    cudaMalloc((void**)&device_b, size * sizeof(float));

    if (host_a == 0 || host_b == 0 || device_a == 0 || device_b == 0) {
        printf("[HOST] Couldn't allocate memory\n");
        return 1;
    }

    cudaError_t err;
    // Populate array a randomly
    fill_array_random(host_a, size);
    err = cudaMemcpy(device_a, host_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(err);

    // Create events for timing
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    cudaCheckError(err);
    err = cudaEventCreate(&stop);
    cudaCheckError(err);

    dim3 blockSize(512);
    dim3 blockCount((size + blockSize.x - 1) / blockSize.x);
    // Launch kernel
    err = cudaEventRecord(start, 0);
    cudaCheckError(err);
    kernel_gflops << <blockCount, blockSize >> > (device_a, device_b);

    float ms = 0;
    err = cudaEventRecord(stop, 0);
    cudaCheckError(err);

    cudaDeviceSynchronize();

    err = cudaEventElapsedTime(&ms, start, stop);
    cudaCheckError(err);
    float seconds = ms / pow((float)10, 3);

    printf("Time: %.5f ms\n", ms);

    /**
     *
     * Set num_ops to the number of floating point operations
     * done in the kernel multiplied with the size of the matrix.
     */

    double num_ops = (double)KERNEL_OPS_COUNT * size;
    double gflops = (double)num_ops / seconds / 1e+9;
    printf("GFLOPS: %.2f\n", gflops);

    free(host_a);
    free(host_b);
    cudaFree(device_a);
    cudaFree(device_b);

    return 0;
}

