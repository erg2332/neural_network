#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>


class Neuron {
public:
  Neuron(int numberOfInputs) : x(numberOfInputs), w(numberOfInputs), a(0.0), y(0.0), delta(0.0) {
    for (int i = 0; i < numberOfInputs; i++) {
      w[i] = static_cast<double> (std::rand()) / RAND_MAX; // random weights in interval [0.0, 1.0]
    }
  }
  void setInput(const std::vector<double> & x) {
    this->x = x;
  }
  void calculateOutput() {
    a = 0.0;
    for (int i = 0; i < x.size(); i++) {
      a += w[i] * x[i];
    }
    y = 1.0 / (1.0 + std::exp(-a));
  }
  void show();
  std::vector<double> x; // inputs
  std::vector<double> w; // weights
  double a;
  double y; // output
  double delta; // for Gradient Descent algorithm
};


class Layer {
public:
  Layer() {}
  Layer(int numberOfNeurons, int numberOfInputs) : neurons(numberOfNeurons, Neuron(numberOfInputs)) {}
  void setInput(const std::vector<double> & x) {
    for (int i = 0; i < neurons.size(); i++) {
      neurons[i].setInput(x);
    }
  }
  void calculateOutput() {
    for (int i = 0; i < neurons.size(); i++) {
      neurons[i].calculateOutput();
    }
  }
  std::vector<double> getOutput() {
    std::vector<double> y(neurons.size());
    for (int i = 0; i < neurons.size(); i++) {
      y[i] = neurons[i].y;
    }
    return y;
  }
  void show();
  std::vector<Neuron> neurons;
};


class NeuralNetwork {
public:
  NeuralNetwork(const std::vector<int> & topology, double learningRate) : layers(topology.size() - 1), eta(learningRate) {
    std::srand(std::time(nullptr));
    for (int i = 0; i < layers.size(); i++) {
      layers[i] = Layer(topology[i + 1], topology[i]);
    }
  }
  void setInput(const std::vector<double> & x) {
    layers[0].setInput(x);
  }
  void calculateOutput() {
    int i = 0;
    int limit = layers.size() - 1;
    while (i < limit) {
      layers[i].calculateOutput();
      layers[i + 1].setInput(layers[i].getOutput());
      i++;
    }
    layers[layers.size() - 1].calculateOutput();
  }
  std::vector<double> getOutput() {
    return layers[layers.size() - 1].getOutput();
  }
  void show();
  double getError(const std::vector<double> & d); // d: desired outputs (to learn) in interval [0.0, 1.0]
  void updateWeightsWithGradientDescentAlgorithm(const std::vector<double> & d);
  std::vector<Layer> layers;
  double eta; // learning rate
private:
  void gradientDescentStep1(const std::vector<double> & d); // calculate delta for each neuron
  void gradientDescentStep2(); // update weights
};


void showVectorDouble(const std::string & label, const std::vector<double> & vv);


int main(int argc, char **argv) {
  const double maximumError = 0.01;
  int iterations = 0;
  double error = 0.0;

  std::vector<int> topology = {4, 5, 6, 3}; // 4 inputs, 5 neurons (layer k=0), 6 neurons (layer k=1), 3 neurons (output layer)

  std::vector<double> x = {5.0, 1.0, 0.0, 3.0};
  std::cout << "INPUT DATA\n";
  showVectorDouble("x", x);

  std::vector<double> d = {1.0, 0.0, 0.5};
  std::cout << "\nDESIRED OUTPUT DATA\n";
  showVectorDouble("d", d);

  std::cout << "\nDESIRED MAXIMUM ERROR : [" << maximumError << "]\n\n";

  NeuralNetwork net(topology, 0.1); // learning rate 0.1
  net.setInput(x);
  net.calculateOutput();

  std::vector<double> y = net.getOutput();
  std::cout << "OUTPUT DATA BEFORE TRAINING\n";
  showVectorDouble("y", y);

  std::cout << "\nTRAINING NEURAL NETWORK...\n";
  while ((error = net.getError(d)) > maximumError) {
    net.updateWeightsWithGradientDescentAlgorithm(d);
    net.calculateOutput();
    iterations++;
  }
  std::cout << "NEURAL NETWORK TRAINED\n\n";

  y = net.getOutput();
  std::cout << "OUTPUT DATA AFTER TRAINING\n";
  showVectorDouble("y", y);

//  std::cout << "\n\n\n";
//  net.show();

  std::cout << "\nERROR : [" << error << "]\n";
  std::cout << "ITERATIONS : [" << iterations << "]\n";

  return 0;
}


void Neuron::show() {
  for (int i = 0; i < x.size(); i++) {
    std::cout << "x(" << i << ") : [" << x[i] << "]\n";
    std::cout << "w(" << i << ") : [" << w[i] << "]\n";
  }
  std::cout << "a : [" << a << "]\n";
  std::cout << "y : [" << y << "]\n";
  std::cout << "delta : [" << delta << "]\n";
}


void Layer::show() {
  for (int i = 0; i < neurons.size(); i++) {
    std::cout << "Neuron " << i << '\n';
    neurons[i].show();
    std::cout << '\n';
  }
}


void NeuralNetwork::show() {
  std::cout << "Learning rate : [" << eta << "]\n\n";
  for (int i = 0; i < layers.size(); i++) {
    std::cout << "LAYER " << i << "\n\n";
    layers[i].show();
    std::cout << '\n';
  }
}


double NeuralNetwork::getError(const std::vector<double> & d) {
  std::vector<double> y(getOutput());
  double error = 0.0;
  for (int i = 0; i < y.size(); i++) {
    error += std::pow(d[i] - y[i], 2.0);
  }
  return error;
}


void NeuralNetwork::updateWeightsWithGradientDescentAlgorithm(const std::vector<double> & d) {
  gradientDescentStep1(d);
  gradientDescentStep2();
}


void NeuralNetwork::gradientDescentStep1(const std::vector<double> & d) {
  // for the last layer
  for (int i = 0; i < layers[layers.size() - 1].neurons.size(); i++) { // neurons of the layer
    layers[layers.size() - 1].neurons[i].delta = 2.0 * (layers[layers.size() - 1].neurons[i].y - d[i]);
  }
  // for the other layers
  int k = layers.size() - 2;
  while (k >= 0) {
    for (int j = 0; j < layers[k].neurons.size(); j++) { // neurons of the layer
      layers[k].neurons[j].delta = 0.0;
      for (int i = 0; i < layers[k + 1].neurons.size(); i++) { // neurons of the next layer
        layers[k].neurons[j].delta += layers[k + 1].neurons[i].delta * layers[k + 1].neurons[i].y * (1.0 - layers[k + 1].neurons[i].y) * layers[k + 1].neurons[i].w[j];
      }
    }
    k--;
  }
}


void NeuralNetwork::gradientDescentStep2() {
  for (int k = 0; k < layers.size(); k++) { // layers
    for (int j = 0; j < layers[k].neurons.size(); j++) { // neurons of the layer
      for (int i = 0; i < layers[k].neurons[j].w.size(); i++) { // weights of the neuron
        layers[k].neurons[j].w[i] -= eta * layers[k].neurons[j].delta * layers[k].neurons[j].y * (1.0 - layers[k].neurons[j].y) * layers[k].neurons[j].x[i];
      }
    }
  }
}


void showVectorDouble(const std::string & label, const std::vector<double> & vv) {
  for (int i = 0; i < vv.size(); i++) {
    std::cout << label << "(" << i << ") : [" << vv[i] << "]\n";
  }
}
