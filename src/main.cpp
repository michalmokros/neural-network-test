#include <iostream>
#include <chrono>
#include "CSVDataReader.hpp"
#include "NeuralNetwork.hpp"

using namespace std;

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

void testmnistdataset() {
    NNInfo nnInfo;
    nnInfo.topology = vector<Topology>{Topology(784, ActivationFunctionType::INPUT), Topology(128, ActivationFunctionType::RELU), Topology(10, ActivationFunctionType::SOFTMAX)};
    NeuralNetwork network(nnInfo);
    CSVDataReader trainData("../data/fashion_mnist_train_vectors.csv", "../data/fashion_mnist_train_labels.csv");

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    std::chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    int a = 1;
    nnweight_t eta = 0.0009;
    nnweight_t alpha = 0.425;
    while (!trainData.isEndOfFile()) {
        ++trainingPass;
        if (trainingPass % a == 0) cout << endl << "Pass " << trainingPass;
        trainData.getNextInputs(inputVals, 0.0, 255.0, 0, 1);
        trainData.getTargetOutputs(targetVals, 10);
        network.trainOnline(inputVals, targetVals, resultVals, eta, alpha);
        if (trainingPass % a == 0) showVectorVals(": Inputs:", inputVals);
        if (trainingPass % a == 0) showVectorVals("Outputs:", resultVals);
        if (trainingPass % a == 0) showVectorVals("Targets:", targetVals);
    }
    
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    cout << endl << "Done" << endl;    
}

void testmnisttrain() {
    NNInfo nnInfo;
    nnInfo.topology = vector<Topology>{Topology(784, ActivationFunctionType::INPUT), Topology(128, ActivationFunctionType::RELU), Topology(10, ActivationFunctionType::SOFTMAX)};
    NeuralNetwork network(nnInfo);
    CSVDataReader trainData("../data/fashion_mnist_train_vectors.csv", "../data/fashion_mnist_train_labels.csv");
    CSVDataReader testData("../data/fashion_mnist_test_vectors.csv");

    vector<vector<nnweight_t>> trainInputVals, trainTargetVals, testInputVals;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    trainData.getAllInputs(trainInputVals, 0.0, 255.0, 0.0, 1.0);
    testData.getAllInputs(testInputVals, 0.0, 255.0, 0.0, 1.0);
    
    trainData.getAllTargetOutputs(trainTargetVals, 10);
    
    nnweight_t eta = 0.001;
    nnweight_t alpha = 0.435;

    vector<nntopology_t> targetVals;
    network.train(trainInputVals, trainTargetVals, testInputVals, targetVals, eta, alpha, 0.1, 1);


    ofstream outfile;
    outfile.open ("../actualPredictions", ios::out | ios::trunc | ios::binary);

    for (auto val : targetVals) {
        outfile << val;
        outfile << endl;
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
}

int main() {
    // testxordataset();
    // testmnistdataset();
    testmnisttrain();
}