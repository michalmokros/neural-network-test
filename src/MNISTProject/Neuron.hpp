#ifndef Included_Neuron_H
#define Included_Neuron_H

#include <vector>
#include "Connection.hpp"
#include "NNTypes.hpp"

using namespace std;

class Neuron {
    public:
        Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex);
        void setOutputValue(nnweight_t outputValue);
        nnweight_t getOutputValue() const;
        void setDeltaWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newDeltaWeight);
        void setWeightOnConnection(const Neuron &connectedNeuron, nnweight_t newWeight);
        nnweight_t getWeightOnConnection(const Neuron &connectedNeuron);
        nnweight_t getWeightOnConnection(size_t connectionIndex);
        nnweight_t getDeltaWeightOnConnection(const Neuron &connectedNeuron);
        void setGradient(nnweight_t gradient);
        nnweight_t getGradient();

        static nnweight_t applicationFunction(nnweight_t x);
        static nnweight_t applicationFunctionDerivationApprox(nnweight_t x);
    private:
        nntopology_t neuronIndex_;
        nnweight_t outputValue_;
        nnweight_t gradient_;
        vector<Connection> outConnections_;
};

#endif