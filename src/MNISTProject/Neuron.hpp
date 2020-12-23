#ifndef Included_Neuron_H
#define Included_Neuron_H

#include <vector>
#include "Connection.hpp"
#include "NNTypes.hpp"

using namespace std;

class Neuron {
    public:
        Neuron(const nntopology_t outputsNumber, const nntopology_t neuronIndex);
        void setOutputValue(const nnweight_t outputValue);
        nnweight_t getOutputValue();
        nnweight_t getWeightOnConnection(const Neuron &connectedNeuron);
        
        static nnweight_t applicationFunction(nnweight_t x);
        static nnweight_t applicationFunctionDerivationApprox(nnweight_t x);
    private:
        nntopology_t neuronIndex_;
        nnweight_t outputValue_;
        vector<Connection> outConnections_;
};

#endif