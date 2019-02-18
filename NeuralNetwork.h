

#ifndef BEGINNING_NEURALNETWORK_H
#define BEGINNING_NEURALNETWORK_H

#include <json/value.h>
#include <ctgmath>
#include <vector>
#include "Neuron.h"

/**********************************************************
 * Program	:  Neural Network
 * Author	:  Braydn Moore
 * Due Date	:  I have lost track of all measures of time so I have zero clue
 * Description	: The main neural network class that handles all the dirty work of getting the output using a feed forward
 *                  algorithm through the network and a back propagation approach to "learn" by interfacing with the neurons
 ***********************************************************/

typedef std::vector<Neuron> Layer;

class NeuralNetwork {
public:
    // constructors for the neural network
    NeuralNetwork(const std::vector<int> topology);
    NeuralNetwork(Json::Value input);
    // feed forward to calculate the output values of the network given the input values
    void feedForward(const std::vector<double> &inputValues);
    // back propagate the neural network using the given target values to adjust the weights using gradients and
    // the root mean squared as our algorithm
    void backPropogation(const std::vector<double> &targetValues);
    // get the output values of the neural network
    void getResults(std::vector<double>& results);
    // convert the neural network to json
    Json::Value toJson();

    // get the error rates for the network
    double getErrorRate() const;
    double getAverageError() const;

private:
    // layers of the network
    std::vector<Layer> layers;
    // private fields for calculating the error rates of the network
    double errorRate, averageError, averageSmoothingFactor;
};


#endif //BEGINNING_NEURALNETWORK_H
