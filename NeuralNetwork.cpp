
#include "NeuralNetwork.h"

// constructor for the network given the topology of the network
NeuralNetwork::NeuralNetwork(const std::vector<int> topology) {
    // zero the error rate
    this->errorRate = 0;
    // get the number of layers for the network
    size_t numLayers = topology.size();

    // for every layer to be created
    for (int layerNumber = 0; layerNumber<numLayers; layerNumber++) {
        // create a new layer
        layers.emplace_back(Layer());

        // add the number of neurons to the newly created layer ensuring there are no connections forward if it is the
        // last layer
        for (int numNeurons = 0; numNeurons<=topology[layerNumber]; numNeurons++)
            layers.back().push_back(Neuron(layerNumber == topology.size()-1?0:topology[layerNumber+1], numNeurons));

        // forces bias of node the node to 1.0
        layers.back().back().setOutputVal(1.0);
    }

}

// intialize a neural network from the json file
NeuralNetwork::NeuralNetwork(Json::Value input) {
    // read in the private fields for the network
    this->errorRate = input["Error Rate"].asDouble();
    this->averageError = input["Average Error"].asDouble();
    this->averageSmoothingFactor = input["Average Smoothing Factor"].asDouble();

    // read in each layer for the network in the json data
    for (Json::Value::ArrayIndex layerIndex = 0; layerIndex!=input["Layers"].size(); layerIndex++){
        // make a new layer
        this->layers.emplace_back(Layer());
        // read in the neuron for the layer for each neuron
        for (Json::Value::ArrayIndex neuronIndex = 0; neuronIndex!=input["Layers"][layerIndex].size(); neuronIndex++)
            this->layers.back().push_back(Neuron(input["Layers"][layerIndex][neuronIndex]));
    }
}

// calculate the results of the neural network given the input
void NeuralNetwork::feedForward(const std::vector<double> &inputValues) {
    // if the input size does not equal the expected size then return
    if (inputValues.size() != layers[0].size() - 1) return;
    // set the input layers output values to be the input into the network
    for (size_t input = 0; input<inputValues.size(); input++)
        layers[0][input].setOutputVal(inputValues[input]);

    // for every neuron in every layer calculate the feed forward information given the output of the previous neuron
    // and the weight of the connection
    for (size_t layerNumber = 1; layerNumber<layers.size(); layerNumber++)
        for (size_t neuronNumber = 0; neuronNumber<layers[layerNumber].size()-1; neuronNumber++)
            layers[layerNumber][neuronNumber].feedForward(layers[layerNumber-1]);

}

// back propagate the neural network by giving it the target values to correct the weights of the connections
void NeuralNetwork::backPropogation(const std::vector<double> &targetValues) {
    // zero the error rate
    this->errorRate = 0;
    // for every neuron in the output layer calculate the error rate using root mean squared of the difference between the
    // expected value and the given value
    for (size_t neuron = 0; neuron<layers.back().size()-1; neuron++)
        errorRate+=pow(targetValues[neuron]-layers.back()[neuron].getOutputVal(), 2);
    errorRate/=layers.back().size()-1;
    errorRate = sqrt(errorRate);

    // calculate running average of error rates for the network to see how well it is performing/learning
    averageError = (averageError*averageSmoothingFactor+errorRate)/(averageSmoothingFactor+1);

    // calculate output layer gradient
    for (size_t neuron = 0; neuron<layers.back().size()-1; neuron++)
        layers.back()[neuron].calculateOutputGradients(targetValues[neuron]);

    // calculate hidden layer gradients
    for (size_t layer = layers.size()-2; layer>0; layer--)
        for (size_t neuron = 0; neuron<layers[layer].size(); neuron++)
            layers[layer][neuron].calculateHiddenGradients(layers[layer+1]);

    // for all layers update connection weight using above gradient data
    for (size_t layer = layers.size()-1; layer>0; layer--)
        for (size_t neuron = 0; neuron<layers[layer].size()-1; neuron++)
            layers[layer][neuron].updateInputWeights(layers[layer-1]);
}

void NeuralNetwork::getResults(std::vector<double> &results) {
    // clear the results vector
    results.clear();
    // get the results of every neuron in the output layer
    for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
        results.push_back(layers.back()[n].getOutputVal());
    }
}

// getters for the error rates of the network
double NeuralNetwork::getErrorRate() const {
    return errorRate;
}

double NeuralNetwork::getAverageError() const {
    return averageError;
}

// convert the network to json
Json::Value NeuralNetwork::toJson() {
    Json::Value ret;
    // set all the private fields of the network to be reused when read back in
    ret["Error Rate"] = this->errorRate;
    ret["Average Error"] = this->averageError;
    ret["Average Smoothing Factor"] = this->averageSmoothingFactor;

    // create the arrays for the neurons and the layers
    Json::Value neuronsInLayer(Json::arrayValue);
    Json::Value jsonLayer(Json::arrayValue);

    // for every layer in the network
    for (size_t layer = 0; layer<layers.size(); layer++){
        // clear the neuron container
        neuronsInLayer.clear();
        // populate the neuron container with the neurons in the layer
        for (size_t neuron = 0; neuron<layers[layer].size(); neuron++)
            neuronsInLayer.append(layers[layer][neuron].toJSON());
        // add the layer to the layer vector
        jsonLayer.append(neuronsInLayer);
    }
    // store the layers
    ret["Layers"] = jsonLayer;
    // return the network as a JSON object
    return ret;
}