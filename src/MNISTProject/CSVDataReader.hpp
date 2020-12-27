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

        void getNextInputs(vector<double> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax);
        void getTargetOutputs(vector<double> &targetOutputVals, nntopology_t size);

    private:
        ifstream trainingDataFile_;
        ifstream labelsDataFile_;
        static void scaler(vector<nnweight_t> &line, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax);
        static vector<nnweight_t> one_hot_encoder(nntopology_t label, nntopology_t size);
};

#endif