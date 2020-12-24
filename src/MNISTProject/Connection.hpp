#ifndef Included_Connection_H
#define Included_Connection_H

#include "NNTypes.hpp"

class Connection {
    public:
        Connection();
        Connection(nnweight_t weight);
        void setWeight(nnweight_t newWeight);
        nnweight_t getWeight();
        void setDeltaWeight(nnweight_t newDeltaWeight);
        nnweight_t getDeltaWeight();
    private:
        nnweight_t weight_;
        nnweight_t deltaWeight_;

        nnweight_t GetRandomWeight(const nnweight_t min, const nnweight_t max);
};

#endif