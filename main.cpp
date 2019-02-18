#include <iostream>
#include <json/writer.h>
#include <json/reader.h>
#include "NeuralNetwork.h"
#include "TrainingData.h"

/**********************************************************
 * Program	:  Basic Neural Network for OOP
 * Author	:  Braydn Moore
 * Due Date	:  I have lost track of all measures of time so I have zero clue
 * Description	: A simple neural network to try to "learn" the boolean operation given that can be read and written.
 *                  The neural network will not print out the exact answers but rather get closer and closer as it trains
 *                  due to the fact it deals with doubles
 ***********************************************************/

 /*
  * DIFFERENCES BETWEEN JAVA AND C++:
  *     1. Typedef: C++ allows you to define custom types which are resolved at compile time, java does not
  *     2. Structs: C++ allows you to declare data structures which are analogous to classes except by default
  *                     access to members of the struct are public by default
  *     3. Operator Overloading: C++ allows you to run custom code in replace of operators between unkown objects
  *     4. Classes: In C++ classes are usually declared in the header along with functions and fields and implemented in
  *                 cpp files whereas java classes are contained to one file
  */

// change this to be any function that returns a boolean for the neural network to learn
std::function<bool (bool, bool)> func = [](bool a, bool b) -> bool{ return (a|b)&(a^b);};
std::string prefix = "mix";


// userful operator overloading for printing out vectors without having to loop every time
template <class T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& vector) {
    os << "[";
    for (auto val : vector)
    {
        os << " " << val;
    }
    os << "]";
    return os;
}

// function to make a neural network and train it using the given data and save it to a given file
void writeNeuralNetwork(std::string output, std::string data){
    // create the file to save the neural network to
    std::fstream neuralNetworkSave;
    // open the file an mark as writing out
    neuralNetworkSave.open(output, std::fstream::out);
    // initialize the training data object
    TrainingData trainingData;
    // read the data into a structure to hold all the data to train the neural network
    NeuralNetworkInput input = trainingData.readTrainingData(data);
    // if the number of input test cases does not equal the number of outputs for the test cases then exit
    if (input.inputs.size() != input.outputs.size()) return;
    // create the network with the given topology from the data read in
    NeuralNetwork network(input.topology);
    // tell them we are training using the data
    std::cout<<"Training"<<std::endl;
    // for every test case
    for (int counter = 0; counter<input.inputs.size(); counter++) {
        // get the input and output vectors and initialize the vector to hold our results
        std::vector<double> inLayer = input.inputs[counter], outLayer = input.outputs[counter];
        std::vector<double> results;
        // if the input or output layer's topology doesn't match the topology of the neural network then exit
        if (inLayer.size() != input.topology[0] || outLayer.size() != input.topology[input.topology.size() - 1])
            return;
        // get the answer by following the neurons and the connections in the network using a feed forward approach
        network.feedForward(inLayer);
        // get the results of the feed forward access
        network.getResults(results);
        // "train" the network by adjusting the weights of the connections
        // by working backwards from the answers the network should have gotten
        network.backPropogation(outLayer);
    }
    // attempt a test run and print out the results
    std::vector<double> results;
    network.feedForward(std::vector<double>({0.0,1.0}));
    network.getResults(results);
    std::cout<<results<<std::endl;
    // write the neural network as a json object to the file
    neuralNetworkSave<<network.toJson();
    // close the file
    neuralNetworkSave.close();
}

// function to read a neural network in from a file and return the network
NeuralNetwork readNeuralNetwork(std::string fileName){
    // network save stream
    std::fstream neuralNetworkSave;
    // open the file and mark it as writing
    neuralNetworkSave.open(fileName, std::fstream::in);
    // if the network cannot be opened return a blank network
    if (!neuralNetworkSave)
        return NeuralNetwork(0);

    // read in the json and convert it to a neural network to be returned
    Json::Value network;
    neuralNetworkSave>>network;
    neuralNetworkSave.close();
    NeuralNetwork test(network);
    return test;
}

// generate training data for the network
void generateTrainingData(std::string fileName){
    TrainingData data;
    data.generateTrainingData(fileName, 100*100*100, func);
}

// test a neural network with static data
void testData(std::string testUnit){
    // read in the neural network
    NeuralNetwork network = readNeuralNetwork(testUnit+".net");
    // make inputs and declare results to store the results of the network
    std::vector<double> input({0.0,1.0});
    std::vector<double> results;
    // feed forward the inputs for the network to calculate the answers and then get the results
    network.feedForward(input);
    network.getResults(results);
    // print out the results of the test data
    std::cout<<results<<std::endl;
}


// do what you will with the main file to test out the neural network
int main() {
    // example of generating a neural network's test data and training/writing network to a file
    generateTrainingData(prefix+".dat");
    writeNeuralNetwork(prefix+".net", prefix+".dat");
    // example of reading in a neural network and using test data to see if it gets the answer right or not
    testData(prefix);
    return 0;
}