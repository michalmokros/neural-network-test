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
    nnInfo.topology = vector<Topology>{Topology(784, ActivationFunctionType::RELU), Topology(64, ActivationFunctionType::RELU), Topology(10, ActivationFunctionType::SOFTMAX)};
    NeuralNetwork network(nnInfo);
    
    CSVDataReader trainData("C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\data\\fashion_mnist_train_vectors.csv", "C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\data\\fashion_mnist_train_labels.csv");
    // CSVDataReader trainData("C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xortraindata.csv", "C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xorlabels.csv");

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    std::chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    
    while (!trainData.isEndOfFile()) {
        ++trainingPass;
        if (trainingPass % 1000 == 0) cout << endl << "Pass " << trainingPass;

        trainData.getNextInputs(inputVals, 0.0, 255.0, 0.05, 0.95);
        if (trainingPass % 1000 == 0) showVectorVals(": Inputs:", inputVals);
        network.feedForward(inputVals);

        network.getResults(resultVals);
        if (trainingPass % 1000 == 0) showVectorVals("Outputs:", resultVals);

        network.getRResults(resultVals);
        if (trainingPass % 1000 == 0) showVectorVals("ROutputs:", resultVals);

        trainData.getTargetOutputs(targetVals, 10);
        if (trainingPass % 1000 == 0) showVectorVals("Targets:", targetVals);

        network.backProp(targetVals);

        if (trainingPass % 1000 == 0) cout << "Net recent average error: " << network.getRecentAverageError() << endl;
    }
    
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    cout << endl << "Done" << endl;
}