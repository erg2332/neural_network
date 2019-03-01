#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"


int main(int argc, char **argv)
{
	double maximum_error = 0.01;
	int iterations;

	std::vector<int> topology;
	topology.push_back(4); // 4 entries (layer k=0)
	topology.push_back(5); // 5 neurons (layer k=1)
	topology.push_back(6); // 6 neurons (layer k=2)
	topology.push_back(3); // 3 neurons (layer k=3)

	std::srand(std::time(NULL));

	neural_network net(topology);

	std::vector<double> x;
	x.push_back(5.0);
	x.push_back(1.0);
	x.push_back(0.0);
	x.push_back(3.0);

	net.setX(x);

	net.calculateY();

	std::vector<double> y = net.getY();

	std::cout << "OUTPUTS\n";
	std::cout << "y[1] : [" << y[0] << "]\n";
	std::cout << "y[2] : [" << y[1] << "]\n";
	std::cout << "y[3] : [" << y[2] << "]\n\n\n";

	std::vector<double> d;
	d.push_back(1.0);
	d.push_back(0.0);
	d.push_back(0.5);

	iterations = 0;
	while (net.getError(d) > maximum_error) {
		net.updateW(d); // backpropagation
		net.calculateY();
		iterations++;
	}

	net.show();

	std::cout << "Error : [" << net.getError(d) << "]\n";
	std::cout << "Iterations : [" << iterations << "]\n";

	return 0;
}
