#ifndef Included_NNInfo_H
#define Included_NNInfo_H

#include <vector>
#include "NNTypes.hpp"

using namespace std;

enum ActivationFunctionType {
    INPUT = 0,
    TANH = 1,
    RELU = 2,
    SOFTMAX = 3
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