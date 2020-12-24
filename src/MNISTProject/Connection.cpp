#include <random>
#include "Connection.hpp"
#include "NNTypes.hpp"

Connection::Connection() {
    weight_ = GetRandomWeight(0.0, 1.0);
}

Connection::Connection(nnweight_t inputWeight) {
    weight_ = inputWeight;
}

nnweight_t Connection::GetRandomWeight(nnweight_t min, nnweight_t max) {
    double randomDouble = (double)rand() / RAND_MAX;
    return min + randomDouble * (max - min);
}

nnweight_t Connection::getWeight() {
    return weight_;
}

nnweight_t Connection::getDeltaWeight() {
    return deltaWeight_;
}

void Connection::setDeltaWeight(nnweight_t newDeltaWeight) {
    deltaWeight_ = newDeltaWeight;
}

void Connection::setWeight(nnweight_t newWeight) {
    weight_ += newWeight;
}
