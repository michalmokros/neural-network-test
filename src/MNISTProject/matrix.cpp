#include "matrix.hpp"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace std;

void matrix::no_corr_result() {
	throw runtime_error("No sane or correct value to return");
}

ostream &operator<<(ostream &out, const matrix &mt) {
	for (size_t row = 0; row < mt.rows(); row++) {
        for (size_t col = 0; col < mt.cols(); col++)
            out << mt[{ row, col }] << ", ";
		out << endl;
	}
	return out;
}

bool operator==(const matrix& leftMatrix, const matrix& rightMatrix) {
    if (leftMatrix.cols() == rightMatrix.cols() && leftMatrix.rows() == rightMatrix.rows()) {
        for (size_t i = 0; i < rightMatrix.rows(); ++i) {
            for (size_t j = 0; j < rightMatrix.cols(); ++j) {
                if (leftMatrix[{i,j}] == rightMatrix[{i,j}]) {
                    continue;
                }
                return false;
            }
        }
        return true;
    }
    return false;
}

bool operator!=(const matrix& leftMatrix, const matrix& rightMatrix) {
    return !(leftMatrix == rightMatrix);
}
