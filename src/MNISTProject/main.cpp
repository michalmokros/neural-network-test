#include "NeuralNetwork.hpp"

using namespace std;

int main() {
    vector<nntopology_t> topology{3, 2, 1};
    NeuralNetwork network(topology);
    
    vector<nnweight_t> inputVals;
    network.feedForward(inputVals);
    
    vector<nnweight_t> targetVals;
    network.backProp(targetVals);

    vector<nnweight_t> resultVals;
    network.getResults(resultVals);
}