/*
 * A matrix composed of consecutive (in memory) elements.
 *
 *  Created on: Oct 25, 2011
 *  Author      : Joao Carlos jcfgonc@gmail.com
 *  License     : MIT License
 */

#ifndef MATRIXF_H_
#define MATRIXF_H_

#include <iostream>
using namespace std;

template<class T>
class Matrix {
private:
public:
	T** matrix; // an array of pointers to arrays
	int ncols;
	int nrows;

	Matrix() {
		this->ncols = 0;
		this->nrows = 0;
		matrix = NULL;
	}

	Matrix(int nrows, int ncols) {
		this->ncols = ncols;
		this->nrows = nrows;
		matrix = new T*[nrows];
		matrix[0] = new T[ncols * nrows]; // the first row contains all the (consecutive) elements of the matrix
		for (int row = 1; row < nrows; row++){
			// some pointer math so that the next rows point to individual regions of the 0th array
			int dx = row * ncols;
			T *m = matrix[0];
			matrix[row] = m + dx;
		}
	}

	~Matrix(){
		delete [] matrix[0];
		delete[] matrix;
	}

	void zero(void) {
		for (int row = 0; row < this->nrows; row++) {
			for (int col = 0; col < this->ncols; col++) {
				matrix[row][col] = 0;
			}
		}
	}

	void setElement(T value, int row, int col) {
		matrix[row][col] = value;
	}

	T getElement(int row, int col) {
		return matrix[row][col];
	}

	T * getElementPointer(int row, int col) {
		return &matrix[row][col];
	}

	void toSTDOUT() {
		for (int row = 0; row < this->nrows; row++) {
			cout << row << ":";
			for (int col = 0; col < this->ncols; col++) {
				cout << matrix[row][col];
				if (col < this->ncols - 1)
					cout << "\t";
			}
			cout << endl;
		}
	}
};

#endif
