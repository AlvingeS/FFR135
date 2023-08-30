CC = g++
CFLAGS = -std=c++17 -Wall

all: main

main: main.o hopfield.o neuron.o
	$(CC) $(CFLAGS) main.o hopfield.o neuron.o -o main

main.o: main.cpp hopfield.h
	$(CC) $(CFLAGS) -c main.cpp

hopfield.o: hopfield.cpp hopfield.h neuron.h
	$(CC) $(CFLAGS) -c hopfield.cpp

neuron.o: neuron.cpp neuron.h
	$(CC) $(CFLAGS) -c neuron.cpp

clean:
	rm -rf *.o main