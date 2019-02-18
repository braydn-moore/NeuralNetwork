
#include "TrainingData.h"

// replace a string in a given string with another string
std::string replace(std::string &str, const std::string &toReplace, const std::string &toAdd) {
    // start position of to replace
    size_t start_pos = 0;
    // while toReplace exists in the string replace it and increment the start position so we find the next one
    while ((start_pos = str.find(toReplace, start_pos))!=std::string::npos) {
        str.replace(start_pos, toReplace.length(), toAdd);
        start_pos+=toAdd.size();
    }
    return str;
}

// split a string along a delimiter
std::vector<std::string> split(const std::string &input, const std::string &search) {
    std::vector<std::string> toReturn;

    // position of last found match of delimiter
    size_t last = 0;
    // index of the next found delimiter after last
    size_t next = 0;
    // while a delimiter still exists in the string split the string
    while ((next = input.find(search, last)) != std::string::npos) {
        toReturn.emplace_back(input.substr(last, next - last));
        last = next + 1;
    }
    // add the remaining piece of the string
    toReturn.emplace_back(input.substr(last));
    // return the vector
    return toReturn;
}

// converts a vector of strings to a vector of ints
std::vector<int> stringVectorToInt(std::vector<std::string> input){
    // initialize the return array
    std::vector<int> ret;
    // convert all the strings in the array into integers using stoi
    std::transform(input.begin(), input.end(), std::back_inserter(ret),
                        [](const std::string& value){return std::stoi(value);});
    // return the array
    return ret;
}

// converts a vector of strings to a vector of doubles
std::vector<double> stringVectorToDouble(std::vector<std::string> input){
    // create the return vector
    std::vector<double> ret;
    // convert all the strings in the vector to doubles using stod
    std::transform(input.begin(), input.end(), std::back_inserter(ret),
                   [](const std::string& value){return std::stod(value);});
    // return the vector of doubles
    return ret;
}

TrainingData::TrainingData() {
    // create a randomization engine using the time since epoch as a seed
    eng = new std::default_random_engine(
            static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()));
    // create a uniform distribution for the engine between 0 and 1(false and true)
    dist = new std::uniform_int_distribution<>(0,1);
}

// generate the training data and write it to a file
void TrainingData::generateTrainingData(std::string fileName, int numSets, std::function<bool(bool, bool)> function) {
    // open the file and mark it as an output
    std::fstream file;
    file.open(fileName, std::fstream::out);
    // set the topology as constant for this current test for boolean
    file<<"Topology: 2,4,1\n";
    // for every test case generate two random booleans and write them as input and then get the output and write to a file
    for (int counter = 0; counter<numSets; counter++){
        int input1 = (*dist)(*eng);
        int input2 = (*dist)(*eng);
        file<<"In: "<<input1<<","<<input2<<"\nOut: "<<function(input1, input2)<<"\n";
    }
    // close the file
    file.close();
}

// get the line of data and convert it to a vector
std::vector<double> TrainingData::getLine(std::string input, std::string toReplace) {
    // replace the input leading info
    input = replace(input, toReplace+": ", "");
    // split the numbers along the , separator and convert them to doubles
    return stringVectorToDouble(split(input, ","));
}

std::vector<int> TrainingData::getTopology(std::string input){
    // get rid of "Topology: " and split along the "," and convert the topology to ints
    input = replace(input, "Topology: ", "");
    return stringVectorToInt(split(input, ","));
}

// read the training data into a neural network input struct
NeuralNetworkInput TrainingData::readTrainingData(std::string fileName) {
    // open the file and initialize the structure to read the data into
    std::fstream file;
    NeuralNetworkInput ret;
    file.open(fileName, std::fstream::in);
    if (!file)
        return ret;

    // get and set the topology in the structure
    std::string topologyString;
    std::getline(file, topologyString);
    ret.topology = getTopology(topologyString);

    // get all the input and output cases and set them to the values in the structure
    std::string line;
    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double >> inputs;
    while (std::getline(file, line)){
        inputs.emplace_back(getLine(line, "In"));
        std::getline(file, line);
        outputs.emplace_back(getLine(line, "Out"));
    }
    ret.inputs = inputs;
    ret.outputs = outputs;
    // close the file and return the structure
    file.close();
    return ret;
}