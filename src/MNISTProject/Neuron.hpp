#ifndef Included_Neuron_H
#define Included_Neuron_H

#include <vector>
#include "Connection.hpp"
#include "NNTypes.hpp"

using namespace std;

class Neuron {
    public:
        Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex);
        Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex, ActivationFunctionType activationFunctionType, nntopology_t previousLayerSize); 
        
        nnweight_t getOutputValue() const;
        void setOutputValue(nnweight_t outputValue);
        
        nnweight_t getGradient() const;
        void setGradient(nnweight_t gradient);

        nnweight_t getWeightOnConnection(const Neuron &connectedNeuron) const;
        nnweight_t getWeightOnConnection(size_t connectionIndex) const;
        void setWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newWeight);
        
        nnweight_t getDeltaWeightOnConnection(const Neuron &connectedNeuron) const;
        void setDeltaWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newDeltaWeight);
        
        static nnweight_t tanhActivationFunction(nnweight_t x);
        static nnweight_t tanhActivationFunctionDerivation(nnweight_t x);
        static nnweight_t reluActivationFunction(nnweight_t x);
        static nnweight_t reluActivationFunctionDerivation(nnweight_t x);
        static nnweight_t softmaxActivationFunction(nnweight_t x, nnweight_t sum);
    
    private:
        nntopology_t neuronIndex_;
        nnweight_t outputValue_;
        nnweight_t gradient_;
        vector<Connection> outConnections_;
};

#endif