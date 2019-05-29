# CUDA Paths.
CUDA_HOME=/usr/local/cuda

# Compilador.
CC=gcc-4.9
CXX=g++-4.9
NVCC=$(CUDA_HOME)/bin/nvcc

# Flags.
FLAGS=-lcuda -lm
CCFLAGS := -I/usr/local/cuda/include
LDFLAGS := -L/usr/local/cuda/lib64

all: compile cuda

compile:
	$(CC) -mcmodel=large -O0 -Wall -fmessage-length=0 -o sincosc sincosc.c -lm

cuda:
	$(NVCC) $(CCFLAGS) $(LDFLAGS) -ccbin $(CXX) sincosc.cu -o sincosc-cuda $(FLAGS)
	
clean:
	rm sincosc sincosc-cuda *.o
