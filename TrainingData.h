

#ifndef NEURALNETWORK_TRAININGDATA_H
#define NEURALNETWORK_TRAININGDATA_H

#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>

/**********************************************************
 * Program	:  Training Data
 * Author	:  Braydn Moore
 * Due Date	:  I have lost track of all measures of time so I have zero clue
 * Description	: A simple class to read generated data from a file into a structure such that it can be easily used to
 *                  train a neural network
 ***********************************************************/

// structure to contain the input data to train a neural network
// used to store the important data read from a file
struct NeuralNetworkInput{
    std::vector<int> topology;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
};

class TrainingData {

public:
    TrainingData();
    void generateTrainingData(std::string fileName, int numSets, std::function<bool(bool, bool)> function);
    NeuralNetworkInput readTrainingData(std::string fileName);

private:
    // used for random number generation for generating training data for the neural network
    std::default_random_engine* eng;
    std::uniform_int_distribution<int>* dist;

    // return a vector of doubles from a line of input in the input file
    std::vector<double> getLine(std::string input, std::string toReplace);
    // gets the topology of a neural network and converts it to a vector
    std::vector<int> getTopology(std::string input);
};


#endif //NEURALNETWORK_TRAININGDATA_H
