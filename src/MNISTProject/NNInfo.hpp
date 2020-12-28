#ifndef Included_NNInfo_H
#define Included_NNInfo_H

#include <vector>
#include "NNTypes.hpp"

using namespace std;

enum ActivationFunctionType {
    TANH = 0,
    RELU = 1,
    SOFTMAX = 2
};

struct Topology {
    Topology(nntopology_t ls, ActivationFunctionType af) { layerSize = ls; activationFunction = af; }
    nntopology_t layerSize;
    ActivationFunctionType activationFunction;
};

struct NNInfo {
    vector<Topology> topology;
};

#endif