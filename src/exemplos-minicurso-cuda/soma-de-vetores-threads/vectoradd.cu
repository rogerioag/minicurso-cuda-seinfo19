#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Host input vectors.
float *h_a;
float *h_b;
// Host output vector.
float *h_c;

// Device input vectors.
float *d_a;
float *d_b;
// Device output vector.
float *d_c;

// Size of arrays.
int n = 0;

/* CUDA kernel. Each thread takes care of one element of c. */
__global__ void vecAdd(float *a, float *b, float *c, int n) {
	int id = threadIdx.x;
	if (id < n)
		c[id] = a[id] + b[id];
}

void init_array() {
	fprintf(stdout, "Inicializando os arrays.\n");
	int i;
	// Initialize vectors on host.
        for (i = 0; i < n; i++) {
		h_a[i] = sinf(i) * sinf(i);
		h_b[i] = cosf(i) * cosf(i);
        }
}

void print_array() {
	int i;
	printf("Imprimindo o Resultado.\n");
	for (i = 0; i < n; i++) {
		fprintf(stdout, "h_c[%07d]: %f\n", i, h_c[i]);
  	}
}

void check_result(){
	// Soma dos elementos do array C e divide por N, o valor deve ser igual a 1.
	int i;
	float sum = 0;
	fprintf(stdout, "Verificando o Resultado.\n");  
	for (i = 0; i < n; i++) {
		sum += h_c[i];
	}
	
	fprintf(stdout, "Resultado Final: (%f, %f)\n", sum, (float)(sum / (float)n));
}

/* Main code */
int main(int argc, char *argv[]) {
	
	if(argc < 2){
		printf("Uso: %s <n>\n", argv[0]);
		exit(0);
	}
	// Size of vectors
	n = atoi(argv[1]);

	printf("Número de Elementos: %d\n", n);

	// Size, in bytes, of each vector
	size_t bytes = n * sizeof(float);
	printf("Memória que será alocada para os 3 arrays: %d\n", 3 * bytes);

	printf("Allocate memory for each vector on host\n");
	// Allocate memory for each vector on host
	h_a = (float *)malloc(bytes);
	h_b = (float *)malloc(bytes);
	h_c = (float *)malloc(bytes);

	printf("Allocate memory for each vector on GPU\n");
	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	printf("Initialize vectors on host\n");
	init_array();

	printf("Copy host vectors to device\n");
	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	printf("Execute the kernel\n");
	vecAdd <<<1,n>>> (d_a, d_b, d_c, n);

	printf("Copy array back to host\n");
	// Copy array back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	print_array();

	check_result();

	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
