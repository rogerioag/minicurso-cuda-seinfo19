/*
 ============================================================================
 Name        : sincosc.c
 Author      : rag@ime.usp.br
 Version     :
 Copyright   : Your copyright notice
 Description : sincos in C.
 ============================================================================
 */
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif
// ./sincos.exe 16 16 16
// 5341.616211 0.552198 1.838819
void sincos_function_(DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* xy, int nx, int ny, int nz) {
	int i, j, k, indice;

	for (i = 0; i < nx; ++i) {
		for (j = 0; j < ny; ++j) {
			for (k = 0; k < nz; ++k) {
				indice = (i * ny * nz) + (j * nz) + k;
				// xy(i, j, k) = sin(x(i, j, k)) + cos(y(i, j, k))
				xy[indice] = sin(x[indice]) + cos(y[indice]);
			}
		}
	}
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

int main(int argc, char* argv[]) {
	
	int nx = 0;
	int ny = 0;
	int nz = 0;

	if (argc != 4) {
		printf("Uso: %s <nx> <ny> <nz>\n", argv[0]);
		return 0;
	}
	else{
		nx = atoi(argv[1]);
		ny = atoi(argv[2]);
		nz = atoi(argv[3]);
		printf("Executando: %s %d %d %d\n", argv[0], nx, ny, nz);
	}

	/* Alocação das estruturas. */
	DATA_TYPE* x = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * nx * ny * nz);
	DATA_TYPE* y = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * nx * ny * nz);
	DATA_TYPE* xy = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * nx * ny * nz);

	/* Inicializa os arrays. */
	init_arrays(x, y, nx, ny, nz);

	sincos_function_(x, y, xy, nx, ny, nz);

	int sizeXy = (nx * ny * nz);

	DATA_TYPE sum = getSum(xy, sizeXy);
	DATA_TYPE min = getMin(xy, sizeXy);
	DATA_TYPE max = getMax(xy, sizeXy);

	printf("sum: %f  min: %f  max: %f\n", sum, min, max);

	free(x);
	free(y);
	free(xy);

	return 0;
}
