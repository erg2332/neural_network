#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_


#include "layer.h"


class neural_network {
public:
	neural_network(std::vector<int> topology);
	void setX(std::vector<double> x);
	void calculateY();
	std::vector<double> getY();
	void show();
	double getError(std::vector<double> d);
	void calculateDelta(std::vector<double> d); // for backpropagation algorithm
	void calculateW(); // for backpropagation algorithm
	void updateW(std::vector<double> d); // for backpropagation algorithm

	std::vector<layer> layers;
	double eta; // for backpropagation algorithm
};


#endif
