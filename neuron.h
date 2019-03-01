#ifndef _NEURON_H_
#define _NEURON_H_


#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>


class neuron {
public:
	neuron(int n);
	void setX(std::vector<double> x);
	void calculateY();
	void show();

	std::vector<double> x;
	std::vector<double> w;
	double a;
	double y;
	double delta; // for backpropagation algorithm
};


#endif
