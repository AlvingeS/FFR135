CC = clang++
CFLAGS = -std=c++17 -Wall -O3 -g -fopenmp
SRC_DIR = src/
OBJ_DIR = obj/

all: main

main: $(OBJ_DIR)main.o $(OBJ_DIR)network.o $(OBJ_DIR)data_processing.o
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)main.o: $(SRC_DIR)main.cpp $(SRC_DIR)network.h $(SRC_DIR)data_processing.h
$(OBJ_DIR)network.o: $(SRC_DIR)network.cpp $(SRC_DIR)network.h $(SRC_DIR)matrix.h
$(OBJ_DIR)data_processing.o: $(SRC_DIR)data_processing.cpp $(SRC_DIR)data_processing.h $(SRC_DIR)matrix.h
$(OBJ_DIR)matrix.o: $(SRC_DIR)matrix.h $(SRC_DIR)vector.h
$(OBJ_DIR)vector.o: $(SRC_DIR)vector.h

clean:
	rm -rf $(OBJ_DIR)*.o main
