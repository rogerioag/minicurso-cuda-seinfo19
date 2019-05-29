#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloKernel() {
	int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    	int threadId = threadIdx.x 
                   + threadIdx.y * blockDim.x
                   + threadIdx.z * blockDim.x * blockDim.y
                   + blockId * blockDim.x * blockDim.y * blockDim.z;
	printf("GPU: gridDim:(%2d, %2d, %2d) blockDim:(%2d, %2d, %2d) blockIdx:(%2d, %2d, %2d) "
         "threadIdx:(%2d, %2d, %2d) -> Thread[%2d]: %s\n", gridDim.x, gridDim.y,
         gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y,
         blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, threadId, "Hello World!!!\n");
}

int main(int argc, char **argv) {

  // Define the GPU id to work.
  cudaSetDevice(0);

  // Hello from host.
  printf("Host: Hello World!!!\n");

  // Hello from GPU.
  helloKernel<<<1,1>>>();

  // Reset device.
  cudaDeviceReset();
  
  return (0);
}
