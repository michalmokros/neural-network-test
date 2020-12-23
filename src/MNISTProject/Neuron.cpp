#include <cmath>
#include "Neuron.hpp"

Neuron::Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex) {
    neuronIndex_ = neuronIndex;
    for (nntopology_t i = 0; i < outputsNumber; i++) {
        outConnections_.push_back(Connection());
    }
}

void Neuron::setOutputValue(const nnweight_t outputValue) {
    outputValue_ = outputValue;
}

nnweight_t Neuron::getOutputValue() {
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
