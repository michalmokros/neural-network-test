#include <cmath>
#include "Neuron.hpp"

Neuron::Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex) {
    neuronIndex_ = neuronIndex;
    for (nntopology_t i = 0; i < outputsNumber; i++) {
        outConnections_.push_back(Connection());
    }
}

void Neuron::setOutputValue(nnweight_t outputValue) {
    outputValue_ = outputValue;
}

nnweight_t Neuron::getOutputValue() const {
    return outputValue_;
}

nnweight_t Neuron::applicationFunction(nnweight_t x) {
    return tanh(x);
}

nnweight_t Neuron::applicationFunctionDerivationApprox(nnweight_t x) {
    return 1.0 - x * x;
}

nnweight_t Neuron::getWeightOnConnection(const Neuron &connectedNeuron) {
    return outConnections_[connectedNeuron.neuronIndex_].getWeight();
}

nnweight_t Neuron::getWeightOnConnection(size_t connectionIndex) {
    return outConnections_[connectionIndex].getWeight();
}

nnweight_t Neuron::getDeltaWeightOnConnection(const Neuron &connectedNeuron) {
    return outConnections_[connectedNeuron.neuronIndex_].getDeltaWeight();
}


void Neuron::setGradient(nnweight_t gradient) {
    gradient_ = gradient;
}

nnweight_t Neuron::getGradient() {
    return gradient_;
}

void Neuron::setDeltaWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newDeltaWeight) {
    outConnections_[connectedNeuron.neuronIndex_].setDeltaWeight(newDeltaWeight);
}

void Neuron::setWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newWeight) {
    outConnections_[connectedNeuron.neuronIndex_].setWeight(newWeight);
}
