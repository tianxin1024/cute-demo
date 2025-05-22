# Directory setup
SRC_DIR := ./src
BIN_DIR := ./bin

# Find all .cu files and generate corresponding executable names
CC_FILES := $(shell find $(SRC_DIR) -name "*.cu")
EXE_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(CC_FILES))

# Ensure bin directory exists
$(shell mkdir -p $(BIN_DIR))

# Default target
all: $(EXE_FILES)
	@echo "Built executables: $(EXE_FILES)"

# Compilation rule
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	@nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include \
		--expt-relaxed-constexpr -cudart shared --cudadevrt none \
		-lcublasLt -lcublas -ldl

test :
	@nvcc -o ./bin/test src/test.cu -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include \
		--expt-relaxed-constexpr -ldl && ./bin/test

# Clean target
clean:
	rm -rf $(BIN_DIR)/*

.PHONY: all clean
