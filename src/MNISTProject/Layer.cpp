#include "Layer.hpp"

Layer::Layer(vector<Neuron> neurons) {
    neurons_ = neurons;
}

void Layer::feedForward(Layer &previousLayer) {
    for (size_t i = 0; i < neurons_.size(); i++) {
        nnweight_t sum = 0.0;
        for (size_t j = 0; j < previousLayer.neurons_.size(); j++) {
            sum += previousLayer.neurons_[j].getOutputValue() * previousLayer.neurons_[j].getWeightOnConnection(neurons_[i]);
        }
    }
}

size_t Layer::neuronsSize() {
    return neurons_.size();
}

void Layer::setNeuronOutputValue(nntopology_t index, nnweight_t outputValue) {
    neurons_[index].setOutputValue(outputValue);
}

