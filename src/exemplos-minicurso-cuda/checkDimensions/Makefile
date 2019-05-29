all: compile

compile: checkDimension.cu dimensions.h
	@echo "Compiling..."	
	nvcc checkDimension.cu -o checkDimension.exe

clean:
	rm *.exe
