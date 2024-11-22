CXX       := g++
CXX_FLAGS := -std=c++17 -ggdb -Iinclude -fopenmp `pkg-config --cflags --libs opencv4`

BIN     := bin
SRC     := src
INCLUDE := include

LIBRARIES   := 
EXECUTABLE  := main # Change if executable wants to be a different name

all: $(BIN)/$(EXECUTABLE)

run: clean all
	@clear
	@./$(BIN)/$(EXECUTABLE)

# Create the bin directory if it doesn't exist, then build the executable
$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp | $(BIN)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $^ -o $@ $(LIBRARIES)

# Rule to create the bin directory if it doesn't exist
$(BIN):
	mkdir -p $(BIN)

clean:
	-rm -rf $(BIN)/*
