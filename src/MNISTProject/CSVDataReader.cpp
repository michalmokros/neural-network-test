#include <sstream>
#include "CSVDataReader.hpp"

CSVDataReader::CSVDataReader(const string inputsFilename, const string targetsFilename) {
    trainingDataFile_.open(inputsFilename);
    labelsDataFile_.open(targetsFilename);
}

void CSVDataReader::getNextInputs(vector<nnweight_t> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    inputVals.clear();

    string line;
    getline(trainingDataFile_, line);
    stringstream ss(line);

    nnweight_t val;
    while (ss >> val) {
        inputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }

    scaler(inputVals, rangeMin, rangeMax, desiredMin, desiredMax);
}

void CSVDataReader::getTargetOutputs(vector<nnweight_t> &targetOutputVals, nntopology_t size) {
    targetOutputVals.clear();

    string line;
    getline(labelsDataFile_, line);
    stringstream ss(line);

    nntopology_t val;
    while (ss >> val) {
        targetOutputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }

    targetOutputVals = one_hot_encoder(targetOutputVals.back(), size);
}

void CSVDataReader::getAllInputs(vector<vector<nnweight_t>> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    while (isEndOfFile()) {
        vector<nnweight_t> iv;
        getNextInputs(iv, rangeMin, rangeMax, desiredMin, desiredMax);
        inputVals.push_back(iv);
    }
}

void CSVDataReader::getAllTargetOutputs(vector<vector<nnweight_t>> &targetOutputVals, nntopology_t size) {
    while (isEndOfFile()) {
        vector<nnweight_t> tv;
        getTargetOutputs(tv, size);
        targetOutputVals.push_back(tv);
    }
}


void CSVDataReader::scaler(vector<nnweight_t> &line, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    for (size_t i = 0; i < line.size(); i++) {
        line[i] = (desiredMax - desiredMin) * (line[i] - rangeMin)/(rangeMax - rangeMin) + desiredMin;
    }
}

vector<nnweight_t> CSVDataReader::one_hot_encoder(nntopology_t label, nntopology_t size) {
    vector<nnweight_t> line;
    for (nntopology_t i = 0; i < size; i++) {
        line.push_back((label == i) ? 1 : 0);
    }

    return line;
}