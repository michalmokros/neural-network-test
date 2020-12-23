#ifndef Included_Layer_H
#define Included_Layer_H

#include <vector>
#include "Neuron.hpp"

using namespace std;

class Layer {
    public:
        Layer(vector<Neuron> neurons);
        void feedForward(Layer &previousLayer);
        size_t layerSize();
        void setNeuronOutputValue(nntopology_t index, nnweight_t outputValue);
        Neuron& getNeuronAt(size_t i);
    private:
        vector<Neuron> neurons_;
};

#endif