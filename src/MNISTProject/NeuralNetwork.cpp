#include <algorithm>
#include <random>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>
#include "NeuralNetwork.hpp"
#include "Neuron.hpp"

using namespace std;

nnweight_t NeuralNetwork::recentAverageFactor_ = 0;

NeuralNetwork::NeuralNetwork(const NNInfo &nninfo) {
    layersSize_ = nninfo.topology.size();
    for (nntopology_t i = 0; i < layersSize_; i++) {
        nntopology_t outputsNumber = i == layersSize_ - 1 ? 0 : nninfo.topology[i + 1].layerSize;

        vector<Neuron> layerVector;
        for (nntopology_t j = 0; j <= nninfo.topology[i].layerSize; j++) {
            if (i == layersSize_ - 1) {
                layerVector.push_back(Neuron(outputsNumber, j));
            } else {
                layerVector.push_back(Neuron(outputsNumber, j, nninfo.topology[i+1].activationFunction, nninfo.topology[i].layerSize, nninfo.topology[i+1].layerSize));
            }
        }

        layers_.push_back(Layer(layerVector, nninfo.topology[i].activationFunction));
    }
}

void NeuralNetwork::trainOnline(const vector<nnweight_t> &inputVals, const vector<nnweight_t> &targetVals, vector<nnweight_t> &resultVals) {
    feedForward(inputVals);
    getResults(resultVals, true);
    backProp(targetVals);
}

void NeuralNetwork::classify(const vector<nnweight_t> &inputVals, vector<nnweight_t> &resultVals) {
    feedForward(inputVals);
    getResults(resultVals, true);
}

nnweight_t NeuralNetwork::train(const vector<vector<nnweight_t>> &inputVals, const vector<vector<nnweight_t>> &targetVals, nnweight_t testRatio, size_t epochs) {
    size_t delimSize = inputVals.size() * (1 - testRatio);
    vector<vector<nnweight_t>> trainX(inputVals.begin(), inputVals.begin() + delimSize);
    vector<vector<nnweight_t>> testX(inputVals.begin() + delimSize, inputVals.end());
    vector<vector<nnweight_t>> trainY(targetVals.begin(), targetVals.begin() + delimSize);
    vector<vector<nnweight_t>> testY(targetVals.begin() + delimSize, targetVals.end());

    vector<size_t> indexes;
    indexes.reserve(trainX.size());
    for (size_t e = 0; e < epochs; e++) {
        shuffleVectorsIndexes(trainX.size(), indexes);
        for (vector<size_t>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1) {
            feedForward(trainX[*it1]);
            backProp(trainY[*it1]);
        }
    }
    vector<nnweight_t> resultVals;
    nnweight_t good = 0;
    nnweight_t bad = 0;
    for (size_t i = 0; i < testX.size(); i++) {
        classify(testX[i], resultVals);
        if (EqualResults(resultVals, testY[i])) {
            good++;
        } else {
            bad++;
        }
    }

    return good / testX.size();
}

bool NeuralNetwork::EqualResults(const vector<nnweight_t> &reslutVals, const vector<nnweight_t> &y) {
    for (size_t i = 0; i < reslutVals.size(); i++) {
        if (reslutVals[i] != y[i]) {
            return false;
        }
    }

    return true;
}

void NeuralNetwork::shuffleVectorsIndexes(size_t size, vector<size_t> &indexes) {
    indexes.clear();

    random_device rd;
    mt19937 g(rd());
    
    for (size_t i = 0; i < size; ++i) {
        indexes.push_back(i);
    }

    shuffle(indexes.begin(), indexes.end(), g);
}

void NeuralNetwork::feedForward(const vector<nnweight_t> &inputVals) {
    if (inputVals.size() != layers_[0].getLayerNeuronsCount() - 1) {
        string message = "Invalid input size: Expected=" + to_string(layers_[0].getLayerNeuronsCount() - 1) + " and Got=" + to_string(inputVals.size());
        throw runtime_error(message);
    }

    for (size_t i = 0; i < inputVals.size(); i++) {
        layers_[0].setNeuronOutputValue(i, inputVals[i]);
    }

    for (nntopology_t i = 1; i < layersSize_; i++) {
        Layer &previousLayer = layers_[i - 1];
        layers_[i].feedForward(previousLayer);
    }
}

void NeuralNetwork::backProp(const vector<nnweight_t> &targetVals) {
    Layer &outputLayer = layers_.back();
    outputLayer.calculateNeuronOutputGradients(targetVals);

    for (size_t i = layers_.size() - 2; i > 0; i--) {
        Layer &hiddenLayer = layers_[i];
        Layer &nextLayer = layers_[i + 1];
        hiddenLayer.calculateHiddenNeuronsGradients(nextLayer);
    }

    for (size_t i = layers_.size() - 1; i > 0; i--) {
        Layer &currentLayer = layers_[i];
        Layer &previousLayer = layers_[i - 1];

        currentLayer.updateNeuronsInputWeights(previousLayer);
    }
}

void NeuralNetwork::getResults(vector<nnweight_t> &resultVals, bool hotEncoded) const {
    resultVals.clear();

    if (hotEncoded) {
        size_t maxIndex = 0;
        nnweight_t maxNum = 0;

        for (size_t i = 0; i < layers_.back().getLayerNeuronsCount() - 1; i++) {
            if (layers_.back().getNeuronOutputValue(i) > maxNum) {
                maxIndex = i;
                maxNum = layers_.back().getNeuronOutputValue(i);
            }
        }

        for (size_t i = 0; i < layers_.back().getLayerNeuronsCount() - 1; i++) {
            if (i == maxIndex) {
                resultVals.push_back(1);
            } else {
                resultVals.push_back(0);
            }
        }
    } else {
        for (size_t i = 0; i < layers_.back().getLayerNeuronsCount() - 1; i++) {
            resultVals.push_back(layers_.back().getNeuronOutputValue(i));
        }
    }
}
