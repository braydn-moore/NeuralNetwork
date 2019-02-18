

#include "Neuron.h"

// the activation function for a neuron used to calculate the output for the feed forward aspect of the neural network
double Neuron::activationFunction(double sum) {
    // hyperbolic tangent function(tanh) which gives an output of a range of -1 to +1 for our activation
    return tanh(sum);
}

// the derivative of the activation function used for calculating the gradients of the neuron for training when
// back propagating
double Neuron::activationFunctionDerivative(double sum) {
    // tanh derivative
    return 1-(pow(sum, 2));
}

// constructor for the neuron
Neuron::Neuron(size_t numOutputs, int index) {
    // set the neuron index in the layer
    this->index = index;
    // make all the connections to the next layer and initialize them with random weights
    for (size_t connectionNumber = 0; connectionNumber<numOutputs; connectionNumber++) {
        outputWeights.emplace_back(Connection());
        outputWeights.back().weight = randomWeight();
    }
}

// neuron constructor from a json value to re construct the value
Neuron::Neuron(Json::Value neuronValue) {
    // re-initialize the the private fields of the neuron from the json value
    this->index = neuronValue["Index"].asInt();
    this->gradient = neuronValue["Gradient"].asDouble();
    this->outputValue = neuronValue["Previous Output Value"].asDouble();
    // if the neuron does not have any connections it is the wrong value so exit the program
    if (!neuronValue["Connections"].isArray()) exit(-1);
    // for every connection of the neuron denoted in the json object add the connection to the output weights
    for (Json::Value::ArrayIndex index = 0; index!=neuronValue["Connections"].size(); index++)
        outputWeights.emplace_back(neuronValue["Connections"][index]);
}

// make a random weight for the neuron
double Neuron::randomWeight() {
    // return a random value between 0 and 1
    return std::rand()/double(RAND_MAX);
}

// getter for the output value of the neuron
double Neuron::getOutputVal() const {
    return outputValue;
}

// setter for the value of the neuron
void Neuron::setOutputVal(double outputValue) {
    Neuron::outputValue = outputValue;
}

// feed the neuron forward
void Neuron::feedForward(const Layer &previousLayer) {
    // initialize the sum of all inputs to the neuron
    double sum = 0.0;

    // for every connecting neuron multiply its value by the weight of the connection
    for (size_t neuronNumber = 0; neuronNumber<previousLayer.size(); neuronNumber++)
        sum += previousLayer[neuronNumber].outputValue * previousLayer[neuronNumber].outputWeights[index].weight;

    // set the output value of the neuron to the activation function of the sum of the neurons
    this->outputValue = Neuron::activationFunction(sum);
}

// get the gradients of the output neuron given the target value
void Neuron::calculateOutputGradients(double target) {
    // gradient function is the difference between the output and the target multiplied by the derivative of the target
    gradient = (target-outputValue)*Neuron::activationFunctionDerivative(outputValue);
}

// get the gradients of any hidden neuron which is any neuron that is not for input or output
void Neuron::calculateHiddenGradients(const Layer &nextLayer) {
    // calculate sum of the derivatives of the next weights of the next layer
    double sumWeights = Neuron::sumOfDerivativeOfNextLayer(nextLayer);
    // the gradient is therefore the sum of the weights times the activation function of the output value
    gradient = sumWeights*Neuron::activationFunctionDerivative(outputValue);
}

// sum the derivative of the next layer
double Neuron::sumOfDerivativeOfNextLayer(const Layer &layer) const {
    // sum value
    double sum = 0;

    // for every neuron in the given layer add the weight times the gradient to the sum
    for (size_t neuron = 0; neuron<layer.size()-1; neuron++)
        sum += outputWeights[neuron].weight * layer[neuron].gradient;

    // return the sum
    return sum;
}

// update the weights of the connections connecting the previous layer to this layer
void Neuron::updateInputWeights(Layer &previousLayer) {
    // for every neuron in the previous layer
    for (size_t neuron = 0; neuron<previousLayer.size(); neuron++){
        Neuron& connectionNeuron = previousLayer[neuron];
        // get the old delta weight
        double oldDelta = connectionNeuron.outputWeights[index].deltaWeight;
        // calculate the new delta weight for the connection
        // alpha = momentum or the magnitude of change of the last update
        double newDelta = (learningRate * connectionNeuron.outputValue * gradient) + (alpha*oldDelta);

        // update the connections information
        connectionNeuron.outputWeights[index].deltaWeight = newDelta;
        connectionNeuron.outputWeights[index].weight+= newDelta;
    }

}

// convert a neuron to a json value
Json::Value Neuron::toJSON() {
    // create the return value
    Json::Value ret;
    // create a connections array
    Json::Value connections(Json::arrayValue);
    // for every connection to a neuron in the next layer add that connection to the json value
    for (Connection connection:outputWeights)
        connections.append(connection.toJSON());

    // store the private fields for the neuron in the json object
    // NOTE: The previous value while shifting does not have to be stored for most neurons however since the bias neurons
    // are not changed during the learning stage of the network their value must be stored so it is not zeroed when we
    // reload the network
    // TODO: Mark bias neurons so we only store their value in the json file
    ret["Previous Output Value"] = outputValue;
    ret["Index"] = index;
    ret["Gradient"] = gradient;
    ret["Connections"] = connections;
    // return the json object
    return ret;
}