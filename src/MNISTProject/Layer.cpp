#include <float.h>
#include <cmath>
#include <stdexcept>
#include "Layer.hpp"

nnweight_t Layer::eta = 0.1;
nnweight_t Layer::alpha = 0.5;

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
    nnweight_t min = DBL_MAX;
    nnweight_t max = DBL_MIN;

    for (size_t i = 0; i < neurons_.size() - 1; i++) {
        nnweight_t sum = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            sum += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }

        switch (activationFunctionType_) {
            case ActivationFunctionType::RELU:
            {
                nnweight_t actVal = Neuron::reluActivationFunction(sum);
                if (actVal > max) max = actVal;
                if (actVal < min) min = actVal;
                neurons_[i].setOutputValue(actVal);
                break;
            }
            case ActivationFunctionType::TANH:
            {
                neurons_[i].setOutputValue(Neuron::tanhActivationFunction(sum));
                break;
            }
            default:
            {
                runtime_error("Not implemented");
                break;
            }
        }
    }

    if (activationFunctionType_ == ActivationFunctionType::RELU) {
        for (size_t i = 0; i < neurons_.size() - 1; i++) {
            neurons_[i].setOutputValue(scaler(neurons_[i].getOutputValue(), min, max, 0, 1));
        }
    }
}

nnweight_t Layer::scaler(nnweight_t val, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax) {
    return (desiredMax - desiredMin) * (val - rangeMin)/(rangeMax - rangeMin) + desiredMin;
}

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

void Layer::updateNeuronsInputWeights(Layer &previousLayer) {
    for (size_t i = 0; i < getLayerNeuronsCount() - 1; i++) {
        updateInputWeights(neurons_[i], previousLayer);
    }
}

void Layer::calculateOutputGradients(Neuron &neuron, const nnweight_t targetVal) {
    switch (activationFunctionType_) {
        case ActivationFunctionType::SOFTMAX:
        {
            neuron.setGradient(targetVal - neuron.getOutputValue());
            break;
        }
        case ActivationFunctionType::TANH:
        {
            nnweight_t delta = targetVal - neuron.getOutputValue();
            nnweight_t grad = delta * Neuron::tanhActivationFunctionDerivation(neuron.getOutputValue());
            neuron.setGradient(grad);
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
        case ActivationFunctionType::TANH:
        {
            nnweight_t sumWeight = sumWeightGradient(neuron, nextLayer);
            nnweight_t grad = sumWeight * Neuron::tanhActivationFunctionDerivation(neuron.getOutputValue());
            neuron.setGradient(grad);
            break;
        }
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

void Layer::updateInputWeights(Neuron &neuron, Layer &previousLayer) {
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
