# gpu-filler Makefile
#
# Build:   make
# Clean:   make clean
# Override CUDA path:  make CUDA_PATH=/opt/cuda

CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
TARGET    := gpu-filler
SRC       := src/gpu_filler.cu

# Cover Volta..Hopper. Tweak if you need older arches.
GENCODE := -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=sm_90

NVCCFLAGS := -O3 -std=c++14 -Xcompiler "-Wall -O3"
INCLUDES  := -I$(CUDA_PATH)/include
LIBDIRS   := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs
LIBS      := -lcublas -lcudart -lnvidia-ml -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(INCLUDES) $(LIBDIRS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
