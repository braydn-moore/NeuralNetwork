

#ifndef BEGINNING_NEURON_H
#define BEGINNING_NEURON_H


#include <vector>
#include <json/json.h>
#include <cstdlib>
#include <ctgmath>
#include <iostream>

/**********************************************************
 * Program	:  Neuron
 * Author	:  Braydn Moore
 * Due Date	:  I have lost track of all measures of time so I have zero clue
 * Description	: The basic building blocks for the network. Used to connect layers where the weights of the connections
 *                  between neurons is the key ingredient to the neural networks "learning"
 ***********************************************************/

// structure of the connection between neurons
struct Connection{
    // the weight and delta weight of the connection
    double weight, deltaWeight;

    // connection constructor passing in the json value to read in the weight and delta weight
    explicit Connection(Json::Value connectionJSON){
        this->weight = connectionJSON["weight"].asDouble();
        this->deltaWeight = connectionJSON["deltaWeight"].asDouble();
    }

    // blank constructor
    Connection() = default;

    // convert the constructor to a JSON object
    Json::Value toJSON(){
        Json::Value value;
        value["weight"] = weight;
        value["deltaWeight"] = deltaWeight;
        return value;
    }
};


// neuron class
class Neuron {
public:
    // define a layer as a vector of neurons
    typedef std::vector<Neuron> Layer;
    // neuron constructors
    Neuron(size_t numOutputs, int index);
    explicit Neuron(Json::Value neuronValue);

    // feed forward values to all connections from the neuron
    void feedForward(const Layer& previousLayer);

    // getters and setters of the output value
    double getOutputVal() const;
    void setOutputVal(double outputValue);

    // calculate the gradients of the neurons to correct the weights of the connection so the network can "learn"
    void calculateOutputGradients(double target);
    void calculateHiddenGradients(const Layer& nextLayer);

    // update the input weights of all the connections between this layer and the next
    void updateInputWeights(Layer& previousLayer);

    // convert the neuron to JSON
    Json::Value toJSON();

private:
    // activation functions and the derivative of the activation function, used for calculating the gradients
    static double activationFunction(double sum);
    static double activationFunctionDerivative(double sum);
    // generate a random weight
    static double randomWeight();
    // learning rate ranges from 0...1
    // alpha ranges from 0...n
    constexpr static double learningRate = 0.15, alpha = 0.5;

    // get the sum of the derivatives of the weights of the next layer
    double sumOfDerivativeOfNextLayer(const Layer& layer) const;

    // the output value and gradient of the current neuron
    double outputValue, gradient;
    // the neuron's index in its layer
    int index;
    // a vector of all the connections between the neuron and all the other neurons in the next layer
    std::vector<Connection> outputWeights;
};


#endif //BEGINNING_NEURON_H
