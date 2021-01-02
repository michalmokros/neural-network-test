#ifndef Included_Connection_H
#define Included_Connection_H

#include "NNInfo.hpp"
#include "NNTypes.hpp"

class Connection {
    public:
        Connection();
        Connection(ActivationFunctionType activationFunctionType, nntopology_t previousLayerSize);

        nnweight_t getWeight() const;
        void setWeight(nnweight_t newWeight);

        nnweight_t getDeltaWeight() const;
        void setDeltaWeight(nnweight_t newDeltaWeight);
   
    private:
        nnweight_t weight_;
        nnweight_t deltaWeight_;

        static nnweight_t GetRandomWeight(const nnweight_t min, const nnweight_t max);
};

#endif