#include <iostream>
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
    vector<nntopology_t> topology{2, 3, 1};
    NeuralNetwork network(topology);
    
    CSVDataReader trainData("C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xortraindata.csv", "C:\\Users\\Martin\\GitProjects\\School\\PV021\\pv021-neural-network\\src\\MNISTProject\\xorlabels.csv");

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEndOfFile()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        trainData.getNextInputs(inputVals);
        showVectorVals(": Inputs:", inputVals);
        network.feedForward(inputVals);

        network.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);

        network.backProp(targetVals);

        cout << "Net recent average error: " << network.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
}