/* Program to compute an embarasingly parallel problem using CUDA
 * Given a vector of numbers, compute the sum of the squares of the numbers
 * The correct result of this is SIZE * 4 = 4000000, since we fill the vector with 2s and 2^2 = 4
*/

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define threadsPerBlock 256
#define SIZE 1000000


inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void computeSum(unsigned char* vector, int* sum) {

    // TODO: Compute the global thread index for this thread. Use blockIdx, blockDim, and threadIdx. It is a 1D grid of 1D blocks, watch slide if not sure how to do this.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes the square of a single element,
    // then adds it to the running total
    if (tid < SIZE) {
        // TODO: Read about atomic operations in CUDA.
        int val = vector[tid];
        atomicAdd(sum, val * val);
    }
}

int main(void) {

    unsigned char* d_vector;
    int* d_sum;
    cudaError_t err;

    err = cudaMalloc((void**)&d_vector, SIZE * sizeof(unsigned char));
    cudaCheckError(err);

    err = cudaMalloc((void**)&d_sum, sizeof(int));
    cudaCheckError(err);

    cudaMemset(d_sum, 0, sizeof(int));
    // Fill the vector with 2s. Each thread will compute the square of a single element, so the sum should be 4 * SIZE = 4000000
    // It is a didactic example, so we are using cudaMemset to fill the vector with 2s.
    cudaMemset(d_vector, 2, SIZE * sizeof(unsigned char));

    dim3 blockSize(threadsPerBlock);
    // If we dont have a multiple of threadsPerBlock, we need to add one more block
    // This is what does (N + K - 1 )/ K
    dim3 blockCount((SIZE + threadsPerBlock - 1) / threadsPerBlock);

    computeSum << <blockCount, blockSize >> > (d_vector, d_sum);
    err = cudaGetLastError();
    cudaCheckError(err);
    int sum;
    err = cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError(err);

    printf("Sum of squares of the vector is %d\n", sum);

    // TODO: d_vector and d_sum are allocated with cudaMalloc but never freed. Free the resources before exiting the program.
    cudaFree(d_vector);
    cudaFree(d_sum);

    return 0;
    // TODO: sum is a 32-bit int. Each byte in the vector is at most 255, so
    // each squared term is at most 65025, and with SIZE = 1000000 elements
    // the total can get big easily and overflow. Try changing the cudaMemset value for
    // d_vector from 2 to 200 and watch what happens to the printed sum.
    // What type should sum actually be to avoid this?
}
