#ifndef Included_Layer_H
#define Included_Layer_H

#include <vector>
#include "Neuron.hpp"

using namespace std;

class Layer {
    public:
        Layer(vector<Neuron> neurons);
        void feedForward(Layer &previousLayer);
        size_t layerSize() const;
        void setNeuronOutputValue(nntopology_t index, nnweight_t outputValue);
        Neuron& getNeuronAt(size_t i);
        void calculateNeuronOutputGradients(const vector<nnweight_t> &targetVals);
        void calculateHiddenNeuronsGradients(Layer &nextLayer);
        void updateNeuronsInputWeights(Layer &previousLayer);
        nnweight_t sumDeltaWeights(Neuron &neuron, Layer &nextLayer);
        nnweight_t getNeuronOutputValue(size_t i) const;

    private:
        vector<Neuron> neurons_;
        static nnweight_t eta;
        static nnweight_t alpha;

        void calculateOutputGradients(Neuron &neuron, const nnweight_t targetVal);
        void calculateHiddenGradients(Neuron &neuron, Layer &nextLayer);
        void updateInputWeights(Neuron &neuron, Layer &previousLayer);

};

#endif