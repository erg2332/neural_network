#!/bin/bash

rm -rf *~

rm -rf net

rm -rf neuron.o
g++ -c neuron.cc

rm -rf layer.o
g++ -c layer.cc

rm -rf neural_network.o
g++ -c neural_network.cc

rm -rf net.o
g++ -c net.cc
g++ -o net net.o neural_network.o layer.o neuron.o

rm -rf neuron.o
rm -rf layer.o
rm -rf neural_network.o
rm -rf net.o

