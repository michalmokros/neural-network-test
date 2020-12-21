#include "matrix.cpp"
#include <iostream>

using namespace std;

int main() {

    const matrix mtOne({ 1, 0, 1, 0,
                     0, 1, 0, 1,
                     1, 0, 1, 0,
                     0, 1, 0, 1 },
        4);

    const matrix mtTwo({ 0, 1, 0, 1,
	                 1, 0, 1, 0,
	                 0, 1, 0, 1,
                     1, 1, 1, 1 },
	    4);

    const int multiplier = 2;

	cout << "************** mtOne **************\n";
	cout << mtOne << endl;

	cout << "************** mtTwo **************\n";
	cout << mtTwo << endl;

	cout << "************** mtOne + mtTwo **************\n";
    cout << (mtOne + mtTwo) << endl;

	cout << "************** mtOne - mtTwo **************\n";
    cout << (mtOne - mtTwo) << endl;

	cout << "************** mtOne * multiplier **************\n";
    cout << (mtOne * multiplier) << endl;

	cout << "************** mtOne[{ 1, 1 }] **************\n";
    cout << "mtOne[{ 1, 1 }]: " << mtOne[{ 1, 1 }] << endl;

	cout << "************* equality comparison **************\n";
	cout << "mtOne == mtOne: " << (mtOne == mtOne) << endl;
	cout << "mtOne != mtTwo: " << (mtOne != mtTwo) << endl;

	const matrix mtThree({ 3, 1, 4 }, 3);

    const matrix mtFour({ 4, 3,
	                 2, 5,
	                 6, 8 },
	    2);
	cout << "************* matrix multiplication **************\n";
	cout << "mtThree * mtFour: " << (mtThree * mtFour) << endl;

	return 0;
}
