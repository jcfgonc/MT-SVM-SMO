/*
 * utils.h
 *
 *  Created on: Oct 26, 2011
 *      Author: jcfgonc@gmail.com
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <string.h>

#ifndef VALUE_SEPARATOR
#define VALUE_SEPARATOR ";\t "
#endif

#ifndef BUFFERING_BLOCK_SIZE
#define BUFFERING_BLOCK_SIZE (1<<20)
#endif

#define FLOAT_COMPARE_TOLERANCE (1E-6)

/**
 * checks if the difference between two floats is within a tolerance
 */
bool fequal(double a, double b, double tolerance) {
	double dif = fabs(a - b);
	return dif < tolerance;
}

/**
 * finds the minimum (first) position in the array where the target occurs
 */
int findMinimumPositionTarget_f(double * array, int array_length, double target) {
	for (int i = 0; i < array_length; i++) {
		double val = array[i];
		if (fequal(val, target, FLOAT_COMPARE_TOLERANCE)) {
			return i;
		}
	}
	return -1;
}

/**
 * finds the minimum (first) position in the array where the target occurs
 */
int findMinimumPositionTarget_i(int * array, int array_length, int target) {
	for (int i = 0; i < array_length; i++) {
		int val = array[i];
		//cout << i << ":" << val << endl;
		if (val == target) {
			return i;
		}
	}
	return -1;
}

/**
 * finds the maximum (last) position in the array where the target occurs
 */
int findMaximumPositionTarget_f(double * array, int array_length, double target) {
	for (int i = array_length - 1; i >= 0; i--) {
		double val = array[i];
		if (fequal(val, target, FLOAT_COMPARE_TOLERANCE)) {
			return i;
		}
	}
	return -1;
}

/**
 * finds the maximum (last) position in the array where the target occurs
 */
int findMaximumPositionTarget_i(int * array, int array_length, int target) {
	for (int i = array_length - 1; i >= 0; i--) {
		int val = array[i];
		if (val == target) {
			return i;
		}
	}
	return -1;
}

/**
 * replaces all occurrences of char 'find' with char 'replace' in given string 'str'
 */
int strrplchr(char * str, char find, char replace) {
	int replacements = 0;
	int position = 0;
	char cur;
	while ((cur = str[position]) != 0) {
		if (cur == find) {
			replacements++;
			str[position] = replace;
		}
		position++;
	}
	return replacements;
}

/**
 * helper function for validLine()
 */
bool validCharacter(char &c) {
	if (c >= '0' && c <= '9')
		return true;
	if (c == ',' || c == '.' || c == ' ' ||c == '\r' || c == '\n' || c == '-' || c == 'e' || c == 'E')
		return true;
	return 0;
}

/**
 * check if a line from the dataset is valid
 * a line is valid if it only contains valid characters :D
 */
bool validLine(char * buf, int size) {
	for (int i = 0; i < size; i++) {
		char c = buf[i];
		if (c == 0)
			return true;
		if (!validCharacter(c))
			return false;
	}
	return true;
}

/**
 * counts the amount of samples in given file
 */int getNumberOfSamples(FILE *f) {
	//TODO: remove that dirty way using valid characters (because of e/E)
	//start... from the beginning
	fseek(f, 0, SEEK_SET);

	//read
	char * buf = new char[BUFFERING_BLOCK_SIZE];
	int count = 0;
	while (fgets(buf, BUFFERING_BLOCK_SIZE, f)) {
		if (validLine(buf, BUFFERING_BLOCK_SIZE))
			count++;
	}
//	if (DEBUG)
//		cout << "Number of samples:\t" << count << endl;
	delete[] buf;
	return count;
}

/**
 * counts the amount of lines in given file
 */
int getNumberOfLines(FILE *f) {
	int count = 0;

	//start from the beginning
	fseek(f, 0, SEEK_SET);

	//read
	char * buf = new char[BUFFERING_BLOCK_SIZE];
	while (1) {
		//read a nice chunk (to minimize head seek overhead)
		size_t amount_read = fread(buf, sizeof(char), BUFFERING_BLOCK_SIZE, f);
		if (amount_read == 0)
			break;
		//count occurrences of '\n' in that chunk
		for (uint i = 0; i < amount_read; i++) {
			if (buf[i] == '\n')
				count++;
		}
	}
	delete[] buf;
//	if (DEBUG)
//		cout << "Number of lines:\t" << count << endl;
	return count;
}

/**
 * counts the amount of columns in first line of given file
 */
int getNumberOfColumns(FILE *f) {
	//start from the beginning
	fseek(f, 0, SEEK_SET);

	//temporary storage
	char * buf = new char[BUFFERING_BLOCK_SIZE];

	//eat empty lines
	bool gotvalidline = false;
	while (!gotvalidline) {
		fgets(buf, BUFFERING_BLOCK_SIZE, f);
		if (buf[0] != '\n' && validLine(buf, BUFFERING_BLOCK_SIZE))
			gotvalidline = true;
	}

	//eat first value
	char* tok = strtok(buf, VALUE_SEPARATOR);
	int num_columns = 1;
	//count next values until the end of the line
	while ((tok = strtok(NULL, VALUE_SEPARATOR)) != NULL)
		num_columns++;

//	if (DEBUG)
//		cout << "Number of columns:\t" << num_columns << endl;
	return num_columns;
}

template<class T>
void readDataSet(FILE *f, Matrix<T> * samples, Array<int> *classes, int ncols, int positive_class) {
	//start... from the beginning
	fseek(f, 0, SEEK_SET);

	//read
	char * buf = new char[BUFFERING_BLOCK_SIZE];
	int row = 0;
	while (fgets(buf, BUFFERING_BLOCK_SIZE, f)) {
		if (!validLine(buf, BUFFERING_BLOCK_SIZE))
			continue;
		//strrplchr(buf, ',', '.'); // replace , by .
		//get first feature and convert to numeric
		char *tok = strtok(buf, VALUE_SEPARATOR);
		double val = strtod(tok, NULL); // atoi IS SLOWER!
		samples->setElement(val, row, 0);
		//do the same for the remaining features
		for (int col = 1; col < ncols - 1; col++) {
			tok = strtok(NULL, VALUE_SEPARATOR);
			val = strtod(tok, NULL);
			// store value
			samples->setElement(val, row, col);
		}
		// get the class
		tok = strtok(NULL, VALUE_SEPARATOR);
		int c = strtol(tok, NULL, 10);
		//we expect the class label to belong to {-1;1}
		if (c != positive_class)
			c = -1;
		//store the class
		classes->setElement(row, c);
		row++;
	}
	if (DEBUG)
		cout << "read dataset with " << row << " rows and " << ncols << " columns" << endl;
}
#endif /* UTILS_H_ */
