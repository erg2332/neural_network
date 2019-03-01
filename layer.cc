#include "layer.h"


// nn: number of neurons
// ni: number of inputs of each neuron
layer::layer(int nn, int ni)
{
	for (int i = 0; i < nn; i++) {
		neurons.push_back(neuron(ni));
	}
}


void layer::setX(std::vector<double> x)
{
	for (int i = 0; i < neurons.size(); i++) {
		neurons[i].setX(x);
	}
}


void layer::calculateY()
{
	for (int i = 0; i < neurons.size(); i++) {
		neurons[i].calculateY();
	}
}


std::vector<double> layer::getY()
{
	std::vector<double> y(neurons.size(), 0.0);
	for (int i = 0; i < neurons.size(); i++) {
		y[i] = neurons[i].y;
	}
	return y;
}


void layer::show()
{
	for (int i = 0; i < neurons.size(); i++) {
		std::cout << "neuron " << i+1 << '\n';
		neurons[i].show();
		std::cout << '\n';
	}
}
