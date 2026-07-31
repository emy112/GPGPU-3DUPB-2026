/* Start program to get the number of CUDA cappable devices
   For those devices, get relevant information as:
    - Device name
    - Compute capability
    - Total global memory
    - Number of SMs(SM = Streaming Multiprocessor)
    - Number of SP cores(SP = Stream Processor)
*/
#include <stdio.h>

// TODO: Have I forgotten to include something ? (Hint: Look at exercise 1 - GLFLOPS)
// Bonus question: Why the program still compiles ?

#include<cuda.h>
#include<cuda_runtime.h>

inline void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {0xa0, 128},
        {0xa1, 128},
        {0xa3, 128},
        {0xb0, 128},
        {0xc0, 128},
        {0xc1, 128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
}

int main() {
    int nDevices;
    cudaError_t err;

    err = cudaGetDeviceCount(&nDevices);
    cudaCheckError(err);

    // The output should be 1 since the only capable one is our main GPU
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Major and minor version of compute capability
    // Those ones can be used to determine the number of cores per SM
    // and the number of SMs per GPU
    int multiProcessorCount = prop.multiProcessorCount;

    // TODO: Modify the code to compute the number of SPs/SM based on the compute capability of your GPU
    // I intentionally hardcoded the number of SPs/SM here
    // Look at the cuda samples from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
    // convert from minor and major version to the number of cores per SM
    int SPsPerSM = _ConvertSMVer2Cores(prop.major, prop.minor);
    int SPcores = multiProcessorCount * SPsPerSM;
    printf("Number of SMs: %d\n", multiProcessorCount);
    printf("Number of SP cores: %d\n", SPcores);

}