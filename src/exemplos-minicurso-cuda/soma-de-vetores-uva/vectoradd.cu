#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Host input vectors.
float *uva_a;
float *uva_b;
// Host output vector.
float *uva_c;

// Size of arrays.
int n = 0;

/* CUDA kernel. Each thread takes care of one element of c. */
__global__ void vecAdd(float *a, float *b, float *c, int n) {
	// Get our global thread ID
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id] + b[id];
}

void init_array() {
	fprintf(stdout, "Inicializando os arrays.\n");
	int i;
	// Initialize vectors on host.
        for (i = 0; i < n; i++) {
		uva_a[i] = sinf(i) * sinf(i);
		uva_b[i] = cosf(i) * cosf(i);
        }
}

void print_array() {
	int i;
	printf("Imprimindo o Resultado.\n");
	for (i = 0; i < n; i++) {
		fprintf(stdout, "uva_c[%07d]: %f\n", i, uva_c[i]);
  	}
}

void check_result(){
	// Soma dos elementos do array C e divide por N, o valor deve ser igual a 1.
	int i;
	float sum = 0;
	fprintf(stdout, "Verificando o Resultado.\n");  
	for (i = 0; i < n; i++) {
		sum += uva_c[i];
	}
	
	fprintf(stdout, "Resultado Final: (%f, %f)\n", sum, (float)(sum / (float)n));
}

/* Main code */
int main(int argc, char *argv[]) {
	// Size of vectors
	n = atoi(argv[1]);

	printf("Número de Elementos: %d\n", n);

	// Size, in bytes, of each vector
	size_t bytes = n * sizeof(float);
	printf("Memória que será alocada para os 3 arrays: %d\n", 3 * bytes);

	printf("Allocate memory for each vector on host\n");
	// Allocate memory for each vector on host
	cudaMallocManaged(&uva_a, bytes);
	cudaMallocManaged(&uva_b, bytes);
	cudaMallocManaged(&uva_c, bytes);

	printf("Initialize vectors on host\n");
	init_array();

	// Number of threads in each thread block.
	int threadsPerBlock = 256;
	// Number of thread blocks in grid.
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	printf("Execute the kernel\n");
	cudaEvent_t start_event, stop_event;
	float time_kernel_execution;
	int eventflags = cudaEventBlockingSync;
	cudaEventCreateWithFlags(&start_event, eventflags);
	cudaEventCreateWithFlags(&stop_event, eventflags);

	/* Recording the time to kernel execution */
	cudaEventRecord(start_event, 0);

	/* Execute the kernel. */
	vecAdd <<< blocksPerGrid, threadsPerBlock >>> (uva_a, uva_b, uva_c, n);

	/* Synchronize */
	cudaDeviceSynchronize();

	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&time_kernel_execution, start_event, stop_event);
	printf("Time Kernel Execution: %f s\n", (time_kernel_execution / 1000.0f));

	print_array();

	check_result();

	printf("Time Kernel Execution: %f ms\n", (time_kernel_execution));

	// Release device memory
	cudaFree(uva_a);
	cudaFree(uva_b);
	cudaFree(uva_c);

	return 0;
}
