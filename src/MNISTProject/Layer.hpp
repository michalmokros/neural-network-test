#ifndef Included_Layer_H
#define Included_Layer_H

#include <vector>
#include "Neuron.hpp"

using namespace std;

class Layer {
    public:
        Layer(vector<Neuron> neurons);
        void feedForward(Layer &previousLayer);
        size_t neuronsSize();
        void setNeuronOutputValue(nntopology_t index, nnweight_t outputValue);

    private:
        vector<Neuron> neurons_;
};

#endif