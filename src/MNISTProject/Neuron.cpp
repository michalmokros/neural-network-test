#include <cmath>
#include <iostream>
#include "Neuron.hpp"

nnweight_t Neuron::reluActivationFunction(nnweight_t x) {
    return x < 0 ? 0 : x;
}

nnweight_t Neuron::reluActivationFunctionDerivation(nnweight_t x) {
    return x < 0 ? 0 : 1;
}

nnweight_t Neuron::softmaxActivationFunction(nnweight_t x, nnweight_t sum) {
    return exp(x) / sum;
}

Neuron::Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex) {
    neuronIndex_ = neuronIndex;
    for (nntopology_t i = 0; i < outputsNumber; i++) {
        outConnections_.push_back(Connection());
    }
}

Neuron::Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex, ActivationFunctionType activationFunctionType, nntopology_t layerFromSize, nntopology_t layerToSize) {
    neuronIndex_ = neuronIndex;
    for (nntopology_t i = 0; i < outputsNumber; i++) {
        outConnections_.push_back(Connection(activationFunctionType, layerFromSize, layerToSize));
    }
}

nnweight_t Neuron::getOutputValue() const {
    return outputValue_;
}

void Neuron::setOutputValue(nnweight_t outputValue) {
    outputValue_ = outputValue;
}

nnweight_t Neuron::getGradient() const {
    return gradient_;
}

void Neuron::setGradient(nnweight_t gradient) {
    gradient_ = gradient;
}

nnweight_t Neuron::getWeightOnConnection(const Neuron &connectedNeuron) const {
    return outConnections_[connectedNeuron.neuronIndex_].getWeight();
}

nnweight_t Neuron::getWeightOnConnection(size_t connectionIndex) const {
    return outConnections_[connectionIndex].getWeight();
}

void Neuron::setWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newWeight) {
    outConnections_[connectedNeuron.neuronIndex_].setWeight(newWeight);
}

nnweight_t Neuron::getDeltaWeightOnConnection(const Neuron &connectedNeuron) const {
    return outConnections_[connectedNeuron.neuronIndex_].getDeltaWeight();
}

nnweight_t Neuron::getDeltaWeightOnConnection(size_t connectionIndex) const {
    return outConnections_[connectionIndex].getDeltaWeight();
}

void Neuron::setDeltaWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newDeltaWeight) {
    outConnections_[connectedNeuron.neuronIndex_].setDeltaWeight(newDeltaWeight);
}