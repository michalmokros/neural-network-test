# Deep Learning from Scratch
**This file may not be up-to-date; read forum, emails, etc.**

## Task:
Implement a feed-forward neural network in C/C++ (or Rust/Swift with limited language support) and train it on a given dataset using a backpropagation algorithm. The dataset is Fashion MNIST [0],  a modern version of a well-known MNIST [1]. Fashion-MNIST is a dataset of Zalando's article images ‒ consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset is in CSV format.

---

## Flow:
- solve the XOR problem first. XOR is a very nice example as a benchmark of the
working learning process with at least one hidden layer. Btw, the presented network
solving XOR in the lecture is minimal and it can be hard to find, so consider
more neurons in the hidden layer. If you can't solve the XOR problem, you can't
solve Fashion MNIST.
- reuse memory. You are implementing an iterative process, so don't allocate new
vectors and matrices all the time. An immutable approach is nice but very
inappropriate. Also ‒ don't copy data in some cache all the time; use indexes.
- objects are fine, but be careful about the depth of object hierarchy you are
going to create. Always remember that you are trying to be fast.
- double precision is fine. You may try to use floats. Do not use BigDecimals or
any other high precision objects.
- simple SGD is not strong, and fast enough, you are going to need to implement some
heuristics as well (or maybe not, but it's highly recommended). I suggest
heuristics: momentum, weight decay, dropout. If you are brave enough, you can
try RMSProp/AdaGrad/Adam.
- start with smaller networks and increase network topology carefully.
- consider validation of the model using part of the train dataset.
- play with hyperparameters to increase your internal validation accuracy.

## Application Structure
src/ (source for the application)
lib/ (source for the application library *.cpp *.hpp)
include/ (interface for the library *.h)
tests/ (main.cpp for quick tests) <- use cppunit for this part
doc/ (doxygen or any kind of documentation)
