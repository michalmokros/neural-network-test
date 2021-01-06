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
        CSVDataReader(const string inputsFilename);
        CSVDataReader(const string inputsFilename, const string targetsFilename);
        bool isEndOfFile() { return trainingDataFile_.eof() || labelsDataFile_.eof(); }

        bool getNextInputs(vector<nnweight_t> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax);
        bool getTargetOutputs(vector<nnweight_t> &targetOutputVals, nntopology_t size);

        void getAllInputs(vector<vector<nnweight_t>> &inputVals, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax);
        void getAllTargetOutputs(vector<vector<nnweight_t>> &targetOutputVals, nntopology_t size);

    private:
        ifstream trainingDataFile_;
        ifstream labelsDataFile_;
        static void scaler(vector<nnweight_t> &line, nnweight_t rangeMin, nnweight_t rangeMax, nnweight_t desiredMin, nnweight_t desiredMax);
        static vector<nnweight_t> one_hot_encoder(nntopology_t label, nntopology_t size);
};

#endif