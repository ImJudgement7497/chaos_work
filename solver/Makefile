# Makefile for compiling with LAPACK and OpenCV

# Compiler and flags
CXX       := g++
CXX_FLAGS := -std=c++17 -ggdb -Iinclude -fopenmp

# LAPACK, LAPACKE, and BLAS libraries
LAPACK_LIBS := -llapacke -llapack -lblas
# OpenCV flags
OPENCV_FLAGS := `pkg-config --cflags --libs opencv4`

# Directories for source, binary, and include files
BIN     := bin
SRC     := src
INCLUDE := include

# Libraries (including OpenCV, LAPACK, LAPACKE, BLAS)
LIBRARIES   := $(OPENCV_FLAGS) $(LAPACK_LIBS)

# Executable name
EXECUTABLE  := main

# Default target
all: $(BIN)/$(EXECUTABLE)

# Run target: clean, build, and run the program
run: clean all
	@clear
	@./$(BIN)/$(EXECUTABLE)

# Compile the source code into the executable
$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp | $(BIN)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $^ -o $@ $(LIBRARIES)

# Create the bin directory if it doesn't exist
$(BIN):
	mkdir -p $(BIN)

# Clean the build directory
clean:
	-rm -rf $(BIN)/*
