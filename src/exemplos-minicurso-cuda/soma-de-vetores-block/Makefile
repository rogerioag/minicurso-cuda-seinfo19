all: compile

compile: vectoradd.cu
	nvcc vectoradd.cu -lcuda -lm -o vectoradd.exe

clean:
	rm *.exe
