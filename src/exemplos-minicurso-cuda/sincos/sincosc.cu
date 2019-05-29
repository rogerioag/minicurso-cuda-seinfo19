#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <math.h>

#include "dimensions.h"

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#define CHECK(call)                                           \
{                                                             \
const cudaError_t error = call;                               \
if (error != cudaSuccess)                                     \
{                                                             \
fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);        \
fprintf(stderr, "code: %d, reason: %s\n", error,              \
cudaGetErrorString(error));                                   \
}                                                             \
}

// ./sincos.exe 16 16 16
// 5341.616211 0.552198 1.838819
void sincos_function_(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz) {
  int i, j, k, indice;
  
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      for (k = 0; k < nz; ++k) {
         indice = (i * ny * nz) + (j * nz) + k;
         xy[indice] = sin(x[indice]) + cos(y[indice]);
      }
    }
  }
}

__global__ void sincos_kernel_3(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz) {
  int i, j, k, indice;
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      for (k = 0; k < nz; ++k) {
        indice = (i * ny * nz) + (j * nz) + k;
        xy[indice] = sin(x[indice]) + cos(y[indice]);
      }
    }
  }
}

/* nx iterações foram transferidas para as dimensões de grid e blocos de threads.
 * 0 <= i < nx, o i é obtido com o id das threads.
 */
__global__ void sincos_kernel_2(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz, int funcId) {
  int i, j, k, indice;
  // i = getGlobalIdx_3D_3D();
  i = getGlobalIdFunc[funcId]();
  // for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      for (k = 0; k < nz; ++k) {
        indice = (i * ny * nz) + (j * nz) + k;
        xy[indice] = sin(x[indice]) + cos(y[indice]);
      }
    }
  //}
}

/* (nx * ny) iterações. Cada thread do arranjo irá executar o laço mais interno somente, o que resulta em 
 * nz iterações.
 * indice = (i * nz) + k
 */
__global__ void sincos_kernel_1(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz, int funcId) {
  int i, k, indice;
  // i = getGlobalIdx_3D_3D();
  i = getGlobalIdFunc[funcId]();
  // for (i = 0; i < nx; ++i) {
  //  for (j = 0; j < ny; ++j) {
  for (k = 0; k < nz; k++) {
  // indice = (i * ny * nz) + (j * nz) + k;
		indice = (i * nz) + k;
		xy[indice] = sin(x[indice]) + cos(y[indice]);
  }
  //  }
  //}
}

__global__ void sincos_kernel_1_unroll_8(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz, int funcId) {
  int i, k, indice;
  // i = getGlobalIdx_3D_3D();
  i = getGlobalIdFunc[funcId]();
  // for (i = 0; i < nx; ++i) {
  //  for (j = 0; j < ny; ++j) {
  for (k = 0; k < nz; k+=8) {
  // indice = (i * ny * nz) + (j * nz) + k;
		indice = (i * nz) + k;
		// xy[indice] = sin(x[indice]) + cos(y[indice]);
		xy[indice] = sin(x[indice]) + cos(y[indice]);
		xy[indice + 1] = sin(x[indice + 1]) + cos(y[indice + 1]);
		xy[indice + 2] = sin(x[indice + 2]) + cos(y[indice + 2]);
		xy[indice + 3] = sin(x[indice + 3]) + cos(y[indice + 3]);
		xy[indice + 4] = sin(x[indice + 4]) + cos(y[indice + 4]);
		xy[indice + 5] = sin(x[indice + 5]) + cos(y[indice + 5]);
		xy[indice + 6] = sin(x[indice + 6]) + cos(y[indice + 6]);
		xy[indice + 7] = sin(x[indice + 7]) + cos(y[indice + 7]);
  }
  //  }
  //}
}

/* Todas as (nx * ny * nz) iterações foram transferidas para as dimensões do arranjo de threads.
 * indice aqui é puro de dimensões.
 * Sendo os ids sequenciais, já irão fazer os acessos de cada iteração.
 * indice = getGlobalIdx_XD_XD();
 */
__global__ void sincos_kernel_0(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz, int funcId) {
  int indice;
  // indice = getGlobalIdx_1D_2D();
  indice = getGlobalIdFunc[funcId]();
  // for (i = 0; i < nx; ++i) {
  //  for (j = 0; j < ny; ++j) {
  //    for (k = 0; k < nz; ++k) {
	//indice = (i * ny * nz) + (j * nz) + k;
  xy[indice] = sin(x[indice]) + cos(y[indice]);
  //    }
  //  }
  //}
}

float getSum(float * xy, int sz) {
  int i;
  float resSum = 0.0;
  for (i = 0; i < sz; i++)
    resSum += xy[i];
  return resSum;
}

float getMin(float * xy, int sz) {
  int i;
  float resMin = xy[0];
  for (i = 1; i < sz; i++)
    if (resMin > xy[i])
      resMin = xy[i];
    return resMin;
}

float getMax(float * xy, int sz) {
  int i;
  float resMax = xy[0];
  for (i = 1; i < sz; i++)
    if (resMax < xy[i])
      resMax = xy[i];
    return resMax;
}

void init_arrays(float* x, float* y, int nx, int ny, int nz){
  int i;
  double invrmax = 1.0 / RAND_MAX;
  for (i = 0; i < nx * ny * nz; i++) {
    x[i] = rand() * invrmax;
    y[i] = rand() * invrmax;
  }
}

int main(int argc, char **argv) {
  int i;
  cudaError_t err;
  int kernel = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int funcId = 0;
  int gpuId = 0;
  
  if (argc != 12) {
    printf("Uso: %s <kernel> <g.x> <g.y> <g.z> <b.x> <b.y> <b.z> <nx> <ny> <nz> <gpuId>\n", argv[0]);

    // funcId agora é calculado conforme as dimensões utilizadas.
    //   printf("Uso: %s <kernel> <g.x> <g.y> <g.z> <b.x> <b.y> <b.z> <nx> <ny> <nz> <funcId> <gpuId>\n", argv[0]);
    //   printf("     funcId:\n");
    //   printf("     0: 1D_1D, 1: 1D_2D, 2: 1D_3D\n");
    //   printf("     3: 2D_1D, 4: 2D_2D, 5: 2D_3D\n");
    //   printf("     6: 3D_1D, 7: 3D_2D, 8: 3D_3D\n");
    return 0;
  }
  else{
    printf("#argumentos (argc): %d\n", argc);
    for (i = 0; i < argc; ++i) {
      printf(" argv[%d]: %s\n", i, argv[i]);
    }
    
    kernel = atoi(argv[1]);
    nx = atoi(argv[8]);
    ny = atoi(argv[9]);
    nz = atoi(argv[10]);
    // funcId = atoi(argv[11]);

		gpuId = atoi(argv[11]);

    printf("Executando: %s sincos_kernel_%d grid(%d, %d, %d) block(%d, %d, %d) %d %d %d\n", argv[0], kernel, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), nx, ny, nz);
  }
  
  /* Recuperar as informações da GPU. */
  printf("%s Starting...\n", argv[0]);
  
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  
  if (deviceCount == 0) {
    printf("Não existem dispositivos com suporte a CUDA.\n");
    return 0;
  } else {
    printf("Existem %d dispositivos com suporte a CUDA.\n", deviceCount);
		if(gpuId > (deviceCount - 1)){
			printf("Não existe um dispositivo sob o id: %d. Utilize %d a %d\n", gpuId, 0, (deviceCount - 1));
			return 0;
		}
  }
  /* Define the gpu id to work */
  cudaSetDevice(gpuId);
  
  /* Alocação das estruturas. */
  // Size, in bytes, of each vector
  size_t bytes = sizeof(DATA_TYPE) * nx * ny * nz;
  printf(" sizeof(DATA_TYPE): %d\n", (int) sizeof(DATA_TYPE));
  size_t totalmem = (3 * bytes);
  printf(" Qtd bytes por estrutura: %zu total: %zu\n", bytes, totalmem);
  
  /* Dados no host. */
  printf("Allocate memory for each vector on host.\n");
  DATA_TYPE *h_x = (DATA_TYPE*) malloc(bytes);
  DATA_TYPE *h_y = (DATA_TYPE*) malloc(bytes);
  DATA_TYPE *h_xy = (DATA_TYPE*) malloc(bytes);
  
  /* Dados no dispositivo. */
  DATA_TYPE *d_x;
  DATA_TYPE *d_y;
  // Device output vector.
  DATA_TYPE *d_xy;
  
  printf("Allocate memory for each vector on GPU.\n");
  // Allocate memory for each vector on GPU
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMalloc(&d_xy, bytes);
  
  /* Inicializa os arrays. */
  init_arrays(h_x, h_y, nx, ny, nz);
  
  printf("Copy host vectors to device.\n");
  // Copy host vectors to device
  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
  
  // Number of threads in each thread block.
  // int threadsPerBlock = 1;
  // Number of thread blocks in grid.
  // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  // int blocksPerGrid = 1;
  
  // dim3 grid(256);            // defines a grid of 256 x 1 x 1 blocks
  // dim3 block(512,512);       // defines a block of 512 x 512 x 1 threads
  
  /* Definição do arranjo de threads em blocos do grid. */
  dim3 grid(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  dim3 block(atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));

  funcId = calculateFunctionId(grid, block);
  printf("funcId: %d\n", funcId);
  
  printf("Execute the kernel.\n");
  cudaEvent_t start_event, stop_event;
  float time_kernel_execution;
  int eventflags = cudaEventBlockingSync;
  cudaEventCreateWithFlags(&start_event, eventflags);
  cudaEventCreateWithFlags(&stop_event, eventflags);
  
  /* Recording the time to kernel execution */
  cudaEventRecord(start_event, 0);
  
  /* Execute the kernel. */
  // sincos_function_(x, y, xy, nx, ny, nz);
  // sincos_kernel_2<<< blocksPerGrid, threadsPerBlock >>>(d_x, d_y, d_xy, nx, ny, nz);
  // sincos_kernel_2<<<dim3(1,1,1), dim3(16,1,1)>>>(d_x, d_y, d_xy, nx, ny, nz);
  
  /* O kernel e a função de calculo do id global são escolhidos conforme o parâmetros.*/
  switch (kernel){
    case 0:
      printf("Executing sincos_kernel_%d.\n", kernel);
      sincos_kernel_0<<<grid, block>>>(d_x, d_y, d_xy, nx, ny, nz, funcId);
      break;
    case 1:
      printf("Executing sincos_kernel_%d.\n", kernel);
      sincos_kernel_1<<<grid, block>>>(d_x, d_y, d_xy, nx, ny, nz, funcId);
      break;
    case 2:
      printf("Executing sincos_kernel_%d.\n", kernel);
      sincos_kernel_2<<<grid, block>>>(d_x, d_y, d_xy, nx, ny, nz, funcId);
      break;
    case 3:
      printf("Executing sincos_kernel_%d.\n", kernel);
      sincos_kernel_3<<<grid, block>>>(d_x, d_y, d_xy, nx, ny, nz);
      break;
    case 4:
      printf("Executing sincos_kernel_%d_unroll_8.\n", kernel);
      sincos_kernel_1_unroll_8<<<grid, block>>>(d_x, d_y, d_xy, nx, ny, nz, funcId);
    default :
      printf("Invalid kernel number.\n");
  }

  err = cudaGetLastError();
  
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
	    cudaGetErrorString(err));
    exit (EXIT_FAILURE);
  }
  /* Synchronize */
  cudaDeviceSynchronize();
  
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&time_kernel_execution, start_event, stop_event);
  printf("Time Kernel Execution: %f s\n", (time_kernel_execution / 1000.0f));
  printf("Time Kernel Execution: %f ms\n", (time_kernel_execution));
  
  printf("Copy array back to host.\n");
  // Copy array back to host
  cudaMemcpy(h_xy, d_xy, bytes, cudaMemcpyDeviceToHost);
  
  int sizeXy = (nx * ny * nz);
  
  DATA_TYPE sum = getSum(h_xy, sizeXy);
  DATA_TYPE min = getMin(h_xy, sizeXy);
  DATA_TYPE max = getMax(h_xy, sizeXy);
  
  printf("sum: %f  min: %f  max: %f\n", sum, min, max);
  
  // Release device memory
  printf("Liberando as estruturas alocadas na Memória da GPU.\n");
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_xy);
  
  printf("Liberando as estruturas alocadas na Memória do host.\n");
  free(h_x);
  free(h_y);
  free(h_xy);
  
  printf("Reset no dispositivo.\n");
  CHECK(cudaDeviceReset());
  
  printf("Done.\n");
  
  return 0;
}
