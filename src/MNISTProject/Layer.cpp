#include <float.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "Layer.hpp"

// eta 0.0025 alpha 0.35 0.81 acc
// eta 0.002 alpha 0.35 0.822167 acc
// eta 0.001 alpha 0.35 0.824167 acc
// eta 0.0005 alpha 0.35 0.818833 acc
// eta 0.0008 alpha 0.35 0.825 acc
// eta 0.0008 alpha 0.25 0.823167 acc
// eta 0.0008 alpha 0.3 0.8235 acc
// eta 0.0008 alpha 0.4 0.825167 acc
// eta 0.0008 alpha 0.5 0.824333
// eta 0.0008 alpha 0.45 0.825167 acc *

// eta 0.0008 alpha 0.45 0.816167 acc topo 784-128-64-10
// eta 0.002 alpha 0.45 0.818333 acc

Layer::Layer(vector<Neuron> &neurons, ActivationFunctionType activationType) {
    neurons_ = neurons;
    activationFunctionType_ = activationType;
    neurons_.back().setOutputValue(1.0);
}

void Layer::feedForward(Layer &previousLayer) {
    switch (activationFunctionType_) {
        case ActivationFunctionType::SOFTMAX:
            softmaxFeedForward(previousLayer);
            break;
        default:
            classicFeedForward(previousLayer);
            break;
    }
}

void Layer::classicFeedForward(Layer &previousLayer) {
    // nnweight_t min = DBL_MAX;
    // nnweight_t max = DBL_MIN;

    for (size_t i = 0; i < neurons_.size() - 1; i++) {
        nnweight_t sum = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            sum += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }

        switch (activationFunctionType_) {
            case ActivationFunctionType::RELU:
            {
                nnweight_t actVal = Neuron::reluActivationFunction(sum);
                // if (actVal > max) max = actVal;
                // if (actVal < min) min = actVal;
                neurons_[i].setOutputValue(actVal);
                break;
            }
            default:
            {
                runtime_error("Not implemented");
                break;
            }
        }
    }
}

// nnweight_t Layer::scaler(nnweight_t val, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
//     return (desiredMax - desiredMin) * (val - rangeMin)/(rangeMax - rangeMin) + desiredMin;
// }

void Layer::softmaxFeedForward(Layer &previousLayer) {
    nnweight_t sumAllWeights = 0.0;
    for (size_t i = 0; i < neurons_.size() - 1; i++) {
        nnweight_t sumOneWeight = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            sumOneWeight += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }
        sumAllWeights += exp(sumOneWeight);
    }

    for (size_t i = 0; i < neurons_.size() - 1; i++) {
        nnweight_t toINeuronSum = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            toINeuronSum += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }

        neurons_[i].setOutputValue(Neuron::softmaxActivationFunction(toINeuronSum, sumAllWeights));
    } 
}

size_t Layer::getLayerNeuronsCount() const {
    return neurons_.size();
}

void Layer::setNeuronOutputValue(nntopology_t index, nnweight_t outputValue) {
    neurons_[index].setOutputValue(outputValue);
}

Neuron& Layer::getNeuronAt(size_t i) {
    return neurons_[i];
}

void Layer::calculateNeuronOutputGradients(const vector<nnweight_t> &targetVals) {
    for (size_t i = 0; i < getLayerNeuronsCount() - 1; i++) {
        calculateOutputGradients(neurons_[i], targetVals[i]);
    }
}

void Layer::calculateHiddenNeuronsGradients(Layer &nextLayer) {
    for (size_t i = 0; i < getLayerNeuronsCount(); i++) {
        calculateHiddenGradients(neurons_[i], nextLayer);
    }
}

void Layer::updateNeuronsInputWeights(Layer &previousLayer, nnweight_t eta, nnweight_t alpha) {
    for (size_t i = 0; i < getLayerNeuronsCount() - 1; i++) {
        updateInputWeights(neurons_[i], previousLayer, eta, alpha);
    }
}

void Layer::calculateOutputGradients(Neuron &neuron, const nnweight_t targetVal) {
    switch (activationFunctionType_) {
        case ActivationFunctionType::SOFTMAX:
        {
            neuron.setGradient(targetVal - neuron.getOutputValue());
            break;
        }
        case ActivationFunctionType::RELU:
        {
            nnweight_t delta = targetVal - neuron.getOutputValue();
            nnweight_t grad = delta * Neuron::reluActivationFunctionDerivation(neuron.getOutputValue());
            neuron.setGradient(grad);
            break;
        }
        default:
        {
            runtime_error("Not implemented");
            break;
        }
    }
}

void Layer::calculateHiddenGradients(Neuron &neuron, Layer &nextLayer) {
    switch (activationFunctionType_) {
        case ActivationFunctionType::RELU:
        {
            nnweight_t sumWeight = sumWeightGradient(neuron, nextLayer);
            nnweight_t grad = sumWeight * Neuron::reluActivationFunctionDerivation(neuron.getOutputValue());
            neuron.setGradient(grad);
            break;
        }
        default:
        {
            runtime_error("Not implemented");
            break;
        }
    }
}

void Layer::updateInputWeights(Neuron &neuron, Layer &previousLayer, nnweight_t eta, nnweight_t alpha) {
    for (size_t i = 0; i < previousLayer.getLayerNeuronsCount(); i++) {
        Neuron &prevNeuron = previousLayer.getNeuronAt(i);
        nnweight_t oldDeltaWeight = prevNeuron.getDeltaWeightOnConnection(neuron);
        nnweight_t newDeltaWeight = eta * prevNeuron.getOutputValue() * neuron.getGradient() + alpha * oldDeltaWeight;
        prevNeuron.setDeltaWeightOnConnection(neuron, newDeltaWeight);
        prevNeuron.setWeightOnConnection(neuron, newDeltaWeight);
    }
}

nnweight_t Layer::sumWeightGradient(Neuron &neuron, Layer &nextLayer) {
    nnweight_t sum = 0.0;
    for (size_t i = 0; i < nextLayer.getLayerNeuronsCount() - 1; i++) {
        sum += neuron.getWeightOnConnection(i) * nextLayer.getNeuronAt(i).getGradient();
    }

    return sum;
}

nnweight_t Layer::getNeuronOutputValue(size_t i) const {
    return neurons_[i].getOutputValue();
}
