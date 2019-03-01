#ifndef _LAYER_H_
#define _LAYER_H_


#include "neuron.h"


class layer {
public:
	layer(int nn, int ni);
	void setX(std::vector<double> x);
	void calculateY();
	std::vector<double> getY();
	void show();

	std::vector<neuron> neurons;
};


#endif
