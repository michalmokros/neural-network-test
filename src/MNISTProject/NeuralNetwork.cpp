#include <algorithm>
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
            layerVector.push_back(Neuron(outputsNumber, j));
        }

        layers_.push_back(Layer(layerVector, nninfo.topology[i].activationFunction));
    }
}

void NeuralNetwork::trainOnline(const vector<nnweight_t> &inputVals, const vector<nnweight_t> &targetVals, vector<nnweight_t> &resultVals) {
    feedForward(inputVals);
    getResults(resultVals);
    backProp(targetVals);
}

void NeuralNetwork::classify(const vector<nnweight_t> &inputVals, vector<nnweight_t> &resultVals) {
    feedForward(inputVals);
    getResults(resultVals);
}

void NeuralNetwork::train(const vector<vector<nnweight_t>> &inputVals, const vector<vector<nnweight_t>> &targetVals) {
    // feedForward(inputVals);
    // backProp(targetVals);
}

void shuffleVectors(const vector<vector<nnweight_t>> &inputVals, const vector<vector<nnweight_t>> &targetVals) {
    // std::vector<int> indexes;
    // indexes.reserve(inputVals.size());
    // for (int i = 0; i < inputVals.size(); ++i)
    //     indexes.push_back(i);
    // shuffle(indexes.begin(), indexes.end());

    // for ( std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1 ) {}
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
    overallNetError_ = 0.0;

    for (size_t i = 0; i < outputLayer.getLayerNeuronsCount() - 1; i++) {
        nnweight_t delta = targetVals[i] - outputLayer.getNeuronAt(i).getOutputValue();
        overallNetError_ += delta * delta;
    } 

    overallNetError_ /= outputLayer.getLayerNeuronsCount() - 1;
    overallNetError_ = sqrt(overallNetError_);

    recentAverageError_ = (recentAverageError_ * recentAverageFactor_ + overallNetError_)
     / (recentAverageFactor_ + 1.0); 

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

void NeuralNetwork::getResults(vector<nnweight_t> &resultVals) const {
    resultVals.clear();

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
}
