#ifndef Included_Layer_H
#define Included_Layer_H

#include <vector>
#include "Neuron.hpp"
#include "NNInfo.hpp"

using namespace std;

class Layer {
    public:
        Layer(vector<Neuron> &neurons, ActivationFunctionType activationType);
        
        void feedForward(Layer &previousLayer);
        void calculateNeuronOutputGradients(const vector<nnweight_t> &targetVals);
        void calculateHiddenNeuronsGradients(Layer &nextLayer);
        void updateNeuronsInputWeights(Layer &previousLayer);

        Neuron& getNeuronAt(size_t i);
        void setNeuronOutputValue(nntopology_t index, nnweight_t outputValue);  
        nnweight_t getNeuronOutputValue(size_t i) const;
        nnweight_t getNeuronROutputValue(size_t i) const;
        size_t getLayerNeuronsCount() const;

    private:
        vector<Neuron> neurons_;
        ActivationFunctionType activationFunctionType_;
        static nnweight_t eta;
        static nnweight_t alpha;

        void calculateOutputGradients(Neuron &neuron, const nnweight_t targetVal);
        void calculateHiddenGradients(Neuron &neuron, Layer &nextLayer);
        void updateInputWeights(Neuron &neuron, Layer &previousLayer);
        nnweight_t sumDeltaWeights(Neuron &neuron, Layer &nextLayer);

        void classicFeedForward(Layer &previousLayer);
        void softmaxFeedForward(Layer &previousLayer);
};

#endif