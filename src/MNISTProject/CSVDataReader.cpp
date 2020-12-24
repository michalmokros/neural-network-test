#include <sstream>
#include "CSVDataReader.hpp"

CSVDataReader::CSVDataReader(const string inputsFilename, const string targetsFilename) {
    trainingDataFile_.open(inputsFilename);
    labelsDataFile_.open(targetsFilename);
}

void CSVDataReader::getNextInputs(vector<double> &inputVals) {
    inputVals.clear();

    string line;
    getline(trainingDataFile_, line);
    stringstream ss(line);

    nnweight_t val;
    while (ss >> val) {
        inputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }
}

void CSVDataReader::getTargetOutputs(vector<double> &targetOutputVals) {
    targetOutputVals.clear();

    string line;
    getline(labelsDataFile_, line);
    stringstream ss(line);

    nnweight_t val;
    while (ss >> val) {
        targetOutputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }
}