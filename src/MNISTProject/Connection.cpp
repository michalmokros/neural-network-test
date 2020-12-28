#include <cstdlib>
#include "Connection.hpp"
#include "NNTypes.hpp"

nnweight_t Connection::GetRandomWeight(nnweight_t min, nnweight_t max) {
    double randomDouble = (double)rand() / RAND_MAX;
    return min + randomDouble * (max - min);
}

Connection::Connection() {
    weight_ = Connection::GetRandomWeight(0.0, 1.0);
}

nnweight_t Connection::getWeight() const {
    return weight_;
}

void Connection::setWeight(nnweight_t newWeight) {
    weight_ += newWeight;
}

nnweight_t Connection::getDeltaWeight() const {
    return deltaWeight_;
}

void Connection::setDeltaWeight(nnweight_t newDeltaWeight) {
    deltaWeight_ = newDeltaWeight;
}
