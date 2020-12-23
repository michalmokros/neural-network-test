#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <iostream>
#include <utility>
#include <vector>
#include <stdexcept>

class matrix {
public:
	using matrix_value_type = double;

private:
	std::vector<matrix_value_type> matrix_;
	short rowsize_;
	short colsize_;

	[[noreturn]] static void no_corr_result();

public:
	matrix(short rowsize, short colsize)
        : matrix_(rowsize * colsize)
	    , rowsize_(rowsize)
	    , colsize_(colsize) {}

	matrix(const std::vector<matrix_value_type> &data, short rowsize)
	    : matrix_(data)
	    , rowsize_(rowsize)
	    , colsize_(matrix_.size() / rowsize_) {}

	// Returns number of rows, number of columns
	std::pair<short, short> size() const {
		return { colsize_, rowsize_ };
	}

	// Returns number of rows
	short rows() const { return colsize_; }

	// Returns number of columns
	short cols() const { return rowsize_; }

    matrix_value_type& operator[](std::pair<short, short> xy) {
        if (xy.first >= rows() || xy.second >= cols()) {
            this->no_corr_result();
        }
        return this->matrix_[rows()*xy.second + xy.first];
    }

    const matrix_value_type& operator[](std::pair<short, short> xy) const {
        if (xy.first >= rows() || xy.second >= cols()) {
            this->no_corr_result();
        }
        return this->matrix_[rows()*xy.second + xy.first];
    }

    matrix& operator+=(const matrix& secondMatrix) {
        if (this->cols() == secondMatrix.cols() && this->rows() == secondMatrix.rows()) {
            for (size_t i = 0; i < secondMatrix.rows(); ++i) {
                for (size_t j = 0; j < secondMatrix.cols(); ++j) {
                    this->matrix_[j*rows() + i] += secondMatrix[{i,j}];
                }
            }
            return *this;
        }
        this->no_corr_result();
    }

    matrix& operator-=(const matrix& secondMatrix) {
        if (this->cols() == secondMatrix.cols() && this->rows() == secondMatrix.rows()) {
            for (size_t i = 0; i < secondMatrix.rows(); ++i) {
                for (size_t j = 0; j < secondMatrix.cols(); ++j) {
                    this->matrix_[j*rows() + i] -= secondMatrix[{i,j}];
                }
            }
            return *this;
        }
        this->no_corr_result();
    }

    matrix& operator*=(int multiplier) {
        for (size_t i = 0; i < this->rows(); ++i) {
            for (size_t j = 0; j < this->cols(); ++j) {
                this->matrix_[j*rows() + i] *= multiplier;
            }
        }
        return *this;
    }

    matrix operator*=(const matrix& secondMatrix) const {
        std::pair<short, short> sMSize = secondMatrix.size();
        if (colsize_ != sMSize.second) {
            throw std::runtime_error("No sane or correct value to return");
        }

        matrix resultingMatrix = matrix(rowsize_, sMSize.first);
        for (int i = 0; i < rowsize_; i++) {
            for (int j = 0; j < sMSize.first; j++) {
                for (int k = 0; k < sMSize.second; k++) {
                    resultingMatrix[{j, i}] += ((*this)[{k, i}] * secondMatrix[{j, k}]);
                }
            }
        }

        return resultingMatrix;
    }
};

std::ostream &operator<<(std::ostream &, const matrix &);

bool operator==(const matrix& leftMatrix, const matrix& rightMatrix);

bool operator!=(const matrix& leftMatrix, const matrix& rightMatrix);

inline matrix& operator+(matrix firstMatrix, const matrix& secondMatrix) {
    return firstMatrix += secondMatrix;
}

inline matrix& operator-(matrix firstMatrix, const matrix& secondMatrix) {
    return firstMatrix -= secondMatrix;
}

inline matrix& operator*(matrix firstMatrix, int multiplier) {
    return firstMatrix *= multiplier;
}

inline matrix& operator*(int multiplier, matrix firstMatrix) {
    return firstMatrix *= multiplier;
}

inline matrix operator*(matrix firstMatrix, matrix secondMatrix) {
    return firstMatrix *= secondMatrix;
    // std::pair<short, short> fMSize = firstMatrix.size();
    // std::pair<short, short> sMSize = secondMatrix.size();
    // if (fMSize.first != sMSize.second) {
	//     throw std::runtime_error("No sane or correct value to return");
    // }

    // matrix resultingMatrix = matrix(fMSize.second, sMSize.first);
    // for (int i = 0; i < fMSize.second; i++) {
    //     for (int j = 0; j < sMSize.first; j++) {
    //         for (int k = 0; k < sMSize.second; k++) {
    //             resultingMatrix[{j, i}] = resultingMatrix[{j, i}] + (firstMatrix[{k, i}] * secondMatrix[{j, k}]);
    //         }
    //     }
    // }

    // return resultingMatrix;
}

#endif
