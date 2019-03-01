#include "neuron.h"


// n: number of inputs
neuron::neuron(int n)
{
	x.resize(n, 0.0);
	w.resize(n, 0.0);
	for (int i = 0; i < x.size(); i++) {
		w[i] = static_cast<double>(std::rand()) / RAND_MAX; // between 0 and 1
	}
	a = 0.0;
	y = 0.0;
	delta = 0.0;
}


void neuron::setX(std::vector<double> x)
{
	this->x = x;
}


void neuron::calculateY()
{
	a = 0.0;
	for (int i = 0; i < x.size(); i++) {
		a += w[i]*x[i];
	}
	y = 1.0 / (1.0 + std::exp(-a));
}


void neuron::show()
{
	for (int i = 0; i < x.size(); i++) {
		std::cout << "x(" << i+1 << ") : [" << x[i] << "]\n";
		std::cout << "w(" << i+1 << ") : [" << w[i] << "]\n";
	}
	std::cout << "a : [" << a << "]\n";
	std::cout << "y : [" << y << "]\n";
	std::cout << "delta : [" << delta << "]\n";
}
