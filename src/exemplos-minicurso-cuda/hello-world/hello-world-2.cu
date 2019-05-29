#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloKernel() {
}

int main(int argc, char **argv) {

  helloKernel<<<1,1>>>();
  printf("Host: Hello World!!!\n");
  
  return (0);
}
