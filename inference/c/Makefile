CC=gcc
CFLAGS=-O3 -Wall -Iinclude

SRC = src/main.c src/model.c src/conv1d.c src/relu.c src/dense.c
OUT = ecg_infer

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)