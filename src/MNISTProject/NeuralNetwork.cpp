#include <iostream>
#include <string>
#include "NeuralNetwork.hpp"
#include "Neuron.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(const vector<nntopology_t> &topology) {
    layersSize_ = topology.size();
    for (nntopology_t i = 0; i < layersSize_; i++) {
        nntopology_t outputsNumber = i == layersSize_ - 1 ? 0 : topology[i + 1];

        vector<Neuron> layerVector;
        for (nntopology_t j = 0; j <= topology[i]; j++) {
            layerVector.push_back(Neuron(outputsNumber, j));
            cout << i << "-" << j << " Neuron created" << endl;
        }

        layers_.push_back(Layer(layerVector));
    }
}

void NeuralNetwork::feedForward(const vector<nnweight_t> &inputVals) {
    if (inputVals.size() != layers_[0].neuronsSize() - 1) {
        string message = "Invalid input size: Expected=" + to_string(layers_[0].neuronsSize() - 1) + " and Got=" + to_string(inputVals.size());
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