main: main.cu aux.cu aux.h Makefile
	nvcc -o main main.cu aux.cu --ptxas-options=-v --use_fast_math -arch sm_35 --compiler-options -Wall -lopencv_highgui -lopencv_core -lcufft

