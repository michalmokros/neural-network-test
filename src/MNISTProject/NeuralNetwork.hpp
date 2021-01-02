#ifndef Included_NeuralNetwork_H
#define Included_NeuralNetwork_H

#include <vector>
#include "Layer.hpp"
#include "NNInfo.hpp"
#include "NNTypes.hpp"

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(const NNInfo &nninfo);
    void trainOnline(const vector<nnweight_t> &inputVals, const vector<nnweight_t> &targetVals, vector<nnweight_t> &resultVals);
    void classify(const vector<nnweight_t> &inputVals, vector<nnweight_t> &resultVals);
    nnweight_t train(const vector<vector<nnweight_t>> &inputVals, const vector<vector<nnweight_t>> &targetVals, nnweight_t testRatio = 0.1, size_t epochs = 1);

private:
    vector<Layer> layers_;
    nntopology_t layersSize_;

    nnweight_t overallNetError_;
    nnweight_t recentAverageError_;
    static nnweight_t recentAverageFactor_;

    void feedForward(const vector<nnweight_t> &inputVals);
    void backProp(const vector<nnweight_t> &targetVals);
    void getResults(vector<nnweight_t> &resultVals, bool hotEncoded) const;
    
    void shuffleVectorsIndexes(size_t size, vector<size_t> &indexes);
    nnweight_t getRecentAverageError() const { return recentAverageError_; }
    bool EqualResults(const vector<nnweight_t> &reslutVals, const vector<nnweight_t> &y);
};

#endif