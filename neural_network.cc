#include "neural_network.h"


neural_network::neural_network(std::vector<int> topology)
{
	for (int i = 1; i < topology.size(); i++) {
		layers.push_back(layer(topology[i], topology[i-1]));
	}
	eta = 0.1;
}


void neural_network::setX(std::vector<double> x)
{
	layers[0].setX(x);
}


void neural_network::calculateY()
{
	for (int i = 0; i < layers.size(); i++) {
		layers[i].calculateY();
		if ((i+1) < layers.size()) {
			layers[i+1].setX(layers[i].getY());
		}
	}
}


std::vector<double> neural_network::getY()
{
	return layers[layers.size()-1].getY();
}


void neural_network::show()
{
	std::cout << "eta : [" << eta << "]\n\n";
	for (int i = 0; i < layers.size(); i++) {
		std::cout << "LAYER " << i+1 << "\n\n";
		layers[i].show();
		std::cout << '\n';
	}
}


double neural_network::getError(std::vector<double> d)
{
	std::vector<double> y(getY());
	double error = 0.0;
	for (int i = 0; i < y.size(); i++) {
		error += std::pow(d[i]-y[i], 2.0);
	}
	return error;
}


// d: desired outputs to learn (between 0 and 1)
void neural_network::calculateDelta(std::vector<double> d)
{
	// last layer
	for (int i = 0; i < layers[layers.size()-1].neurons.size(); i++) {
		layers[layers.size()-1].neurons[i].delta = 2.0*(layers[layers.size()-1].neurons[i].y - d[i]);
	}

	// for the other layers
	for (int k = (layers.size()-2); k >= 0; k--) { // layers
		for (int j = 0; j < layers[k].neurons.size(); j++) { // neurons of the layer
			layers[k].neurons[j].delta = 0.0;
			for (int i = 0; i < layers[k+1].neurons.size(); i++) { // neurons of the next layer
				layers[k].neurons[j].delta += layers[k+1].neurons[i].delta * layers[k+1].neurons[i].y * (1.0-layers[k+1].neurons[i].y) * layers[k+1].neurons[i].w[j];
			}
		}
	}
}


void neural_network::calculateW()
{
	for (int k = 0; k < layers.size(); k++) { // layers
		for (int j = 0; j < layers[k].neurons.size(); j++) { // neurons of the layer
			for (int i = 0; i < layers[k].neurons[j].w.size(); i++) { // weights of the neuron
				if (k > 1) {
					layers[k].neurons[j].w[i] -= eta * layers[k].neurons[j].delta * layers[k].neurons[j].y * (1.0-layers[k].neurons[j].y) * layers[k-1].neurons[i].y;
				} else {
					layers[k].neurons[j].w[i] -= eta * layers[k].neurons[j].delta * layers[k].neurons[j].y * (1.0-layers[k].neurons[j].y) * layers[k].neurons[j].x[i];
				}
			}
		}
	}
}


void neural_network::updateW(std::vector<double> d)
{
	calculateDelta(d);
	calculateW();
}
