#include <chrono>
#include <iostream>
#include <sstream>
#include "CSVDataReader.hpp"

CSVDataReader::CSVDataReader(const string inputsFilename) {
    trainingDataFile_.open(inputsFilename);
}

CSVDataReader::CSVDataReader(const string inputsFilename, const string targetsFilename) {
    trainingDataFile_.open(inputsFilename);
    labelsDataFile_.open(targetsFilename);
}

bool CSVDataReader::getNextInputs(vector<nnweight_t> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    inputVals.clear();

    string line;
    getline(trainingDataFile_, line);
    if (line.empty()) {
        return false;
    }
    stringstream ss(line);
    nnweight_t val;
    while (ss >> val) {
        inputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }

    scaler(inputVals, rangeMin, rangeMax, desiredMin, desiredMax);
    return true;
}

bool CSVDataReader::getTargetOutputs(vector<nnweight_t> &targetOutputVals, nntopology_t size) {
    targetOutputVals.clear();

    string line;
    getline(labelsDataFile_, line);
    if (line.empty()) {
        return false;
    }
    stringstream ss(line);

    nntopology_t val;
    while (ss >> val) {
        targetOutputVals.push_back(val);
        if(ss.peek() == ',') ss.ignore();
    }

    targetOutputVals = one_hot_encoder(targetOutputVals.back(), size);
    return true;
}

void CSVDataReader::getAllInputs(vector<vector<nnweight_t>> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    int l = 0;
 
    while (trainingDataFile_) {
        l++;
        string s;
        if (!getline(trainingDataFile_, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            vector<double> record;
 
            while (ss) {
                string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    cout << "NaN found in file " << endl;
                    e.what();
                }
            }
 
            scaler(record, rangeMin, rangeMax, desiredMin, desiredMax);
            inputVals.push_back(record);    
        }
    }
 
    if (!trainingDataFile_.eof()) {
        cerr << "Could not read file " << endl;
    }
}

void CSVDataReader::getAllTargetOutputs(vector<vector<nnweight_t>> &targetOutputVals, nntopology_t size) {
    int l = 0;
 
    while (labelsDataFile_) {
        l++;
        string s;
        if (!getline(labelsDataFile_, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            vector<double> record;
 
            while (ss) {
                string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    cout << "NaN found in file " << endl;
                    e.what();
                }
            }
 
            record = one_hot_encoder(record.back(), size);
            targetOutputVals.push_back(record);    
        }
    }
 
    if (!trainingDataFile_.eof()) {
        cerr << "Could not read file " << endl;
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