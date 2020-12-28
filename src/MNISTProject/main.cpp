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

int main() {
    NNInfo nnInfo;
    // nnInfo.topology = vector<Topology>{Topology(784, ActivationFunctionType::RELU), Topology(64, ActivationFunctionType::RELU), Topology(10, ActivationFunctionType::SOFTMAX)};
    nnInfo.topology = vector<Topology>{Topology(2, ActivationFunctionType::TANH), Topology(4, ActivationFunctionType::TANH), Topology(2, ActivationFunctionType::TANH)};

    NeuralNetwork network(nnInfo);
    
    // CSVDataReader trainData("C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\data\\fashion_mnist_train_vectors.csv", "C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\data\\fashion_mnist_train_labels.csv");
    CSVDataReader trainData("C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xortraindata.csv", "C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xorlabels.csv");

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    std::chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    
    while (!trainData.isEndOfFile()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        trainData.getNextInputs(inputVals, 0.0, 1.0, 0, 1);
        showVectorVals(": Inputs:", inputVals);
        network.feedForward(inputVals);

        network.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // network.getRResults(resultVals);
        // showVectorVals("ROutputs:", resultVals);

        trainData.getTargetOutputs(targetVals, 2);
        showVectorVals("Targets:", targetVals);

        network.backProp(targetVals);

        cout << "Net recent average error: " << network.getRecentAverageError() << endl;
    }
    
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    cout << endl << "Done" << endl;
}