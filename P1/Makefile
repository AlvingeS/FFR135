CC = clang++
CFLAGS = -std=c++17 -Wall
SRC_DIR = src/
OBJ_DIR = obj/

all: main

main: $(OBJ_DIR)main.o $(OBJ_DIR)hopfield_network.o $(OBJ_DIR)utils.o $(OBJ_DIR)neuron.o
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)main.o: $(SRC_DIR)main.cpp $(SRC_DIR)hopfield_network.h $(SRC_DIR)utils.h
$(OBJ_DIR)hopfield_network.o: $(SRC_DIR)hopfield_network.cpp $(SRC_DIR)hopfield_network.h $(SRC_DIR)utils.h $(SRC_DIR)neuron.h
$(OBJ_DIR)utils.o: $(SRC_DIR)utils.cpp $(SRC_DIR)utils.h
$(OBJ_DIR)neuron.o: $(SRC_DIR)neuron.cpp $(SRC_DIR)neuron.h

clean:
	rm -rf $(OBJ_DIR)*.o main
