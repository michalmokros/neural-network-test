#include <vector>
#ifndef Included_CSVDataReader_H
#define Included_CSVDataReader_H

#include <fstream>
#include <string>
#include "NNTypes.hpp"

using namespace std;

class CSVDataReader
{
    public:
        CSVDataReader(const string inputsFilename, const string targetsFilename);
        bool isEndOfFile() { return trainingDataFile_.eof() || labelsDataFile_.eof(); }

        void getNextInputs(vector<double> &inputVals);
        void getTargetOutputs(vector<double> &targetOutputVals);

    private:
        ifstream trainingDataFile_;
        ifstream labelsDataFile_;
};

#endif