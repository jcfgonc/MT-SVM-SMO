/*
 * Array.h
 *
 *  Created on: Oct 25, 2011
 *  Author      : Joao Carlos jcfgonc@gmail.com
 *  License     : MIT License
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <iostream>
using namespace std;

template<class T>
class Array {
public:
	unsigned int size;
	T*array;

	Array(unsigned int _size) {
		this->size = _size;
		array = new T[_size];
	}

	Array() {
		this->size = 0;
		array = NULL;
	}

	void setElement(unsigned int position, T value) {
		array[position] = value;
	}

	void zero() {
		for (unsigned int pos = 0; pos < this->size; pos++) {
			array[pos] = 0;
		}
	}

	T getElement(unsigned int position) {
		return array[position];
	}

	T * getElementPointer(unsigned int position) {
		return &array[position];
	}

	void toSTDOUT() {
		for (unsigned int pos = 0; pos < this->size; pos++) {
			cout << "[" << pos << "]:" << array[pos] << ",";
		}
		cout << endl;
	}

	friend ostream& operator <<(ostream &out, Array<T> const &e) {
		for (unsigned int pos = 0; pos < e.size; pos++) {
			cout << "," << e.array[pos];
		}
		return out;
	}
};
#endif /* ARRAY_H_ */
