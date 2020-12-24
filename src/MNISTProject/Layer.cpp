#include "Layer.hpp"

nnweight_t Layer::eta = 0.15;
nnweight_t Layer::alpha = 0.5;

Layer::Layer(vector<Neuron> neurons) {
    neurons_ = neurons;
}

void Layer::feedForward(Layer &previousLayer) {
    for (size_t i = 0; i < neurons_.size(); i++) {
        nnweight_t sum = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            sum += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }

        neurons_[i].setOutputValue(Neuron::applicationFunction(sum));
    }
}

size_t Layer::layerSize() const {
    return neurons_.size();
}

void Layer::setNeuronOutputValue(nntopology_t index, nnweight_t outputValue) {
    neurons_[index].setOutputValue(outputValue);
}

Neuron& Layer::getNeuronAt(size_t i) {
    return neurons_[i];
}

void Layer::calculateNeuronOutputGradients(const vector<nnweight_t> &targetVals) {
    for (size_t i = 0; i < layerSize() - 1; i++) {
        calculateOutputGradients(getNeuronAt(i), targetVals[i]);
    }
}

void Layer::calculateHiddenNeuronsGradients(Layer &nextLayer) {
    for (size_t i = 0; i < layerSize(); i++) {
        calculateHiddenGradients(getNeuronAt(i), nextLayer);
    }
}

void Layer::updateNeuronsInputWeights(Layer &previousLayer) {
    for (size_t i = 0; i < layerSize(); i++) {
        updateInputWeights(getNeuronAt(i), previousLayer);
    }
}


void Layer::calculateOutputGradients(Neuron &neuron, const nnweight_t targetVal) {
    double delta = targetVal - neuron.getOutputValue();
    double grad = delta * Neuron::applicationFunctionDerivationApprox(neuron.getOutputValue());
    neuron.setGradient(grad);
}

void Layer::calculateHiddenGradients(Neuron &neuron, Layer &nextLayer) {
    nnweight_t deltaWeightsSum = sumDeltaWeights(neuron, nextLayer);
    double grad = deltaWeightsSum * Neuron::applicationFunctionDerivationApprox(neuron.getOutputValue());
    neuron.setGradient(grad);
}

void Layer::updateInputWeights(Neuron &neuron, Layer &previousLayer) {
    for (size_t i = 0; i < previousLayer.layerSize(); i++) {
        Neuron &prevNeuron = previousLayer.getNeuronAt(i);
        nnweight_t oldDeltaWeight = prevNeuron.getDeltaWeightOnConnection(neuron);
        nnweight_t newDeltaWeight = eta * prevNeuron.getOutputValue() * neuron.getGradient() * alpha * oldDeltaWeight;
        prevNeuron.setDeltaWeightOnConnection(neuron, newDeltaWeight);
        prevNeuron.setWeightOnConnection(neuron, newDeltaWeight);
    }
}

nnweight_t Layer::sumDeltaWeights(Neuron &neuron, Layer &nextLayer) {
    nnweight_t sum = 0.0;
    for (size_t i = 0; i < nextLayer.layerSize() - 1; i++) {
        sum += neuron.getWeightOnConnection(i) * nextLayer.getNeuronAt(i).getGradient();
    }

    return sum;
}

nnweight_t Layer::getNeuronOutputValue(size_t i) const {
    return neurons_[i].getOutputValue();
}
