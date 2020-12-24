#ifndef Included_NeuralNetwork_H
#define Included_NeuralNetwork_H

#include <vector>
#include "Layer.hpp"
#include "NNTypes.hpp"

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(const vector<nntopology_t> &topology);
    void feedForward(const vector<nnweight_t> &inputVals);
    void backProp(const vector<nnweight_t> &targetVals);
    void getResults(vector<nnweight_t> &resultVals) const;
private:
    vector<Layer> layers_;
    nntopology_t layersSize_;

    nnweight_t overallNetError_;
    nnweight_t recentAverageError_;
    static nnweight_t recentAverageFactor_;
};

#endif