/*
 * smo.h
 *
 *  Created on: Oct 26, 2011
 *      Author: ck
 */

#ifndef SMO_H_
#define SMO_H_

#include <iostream>
#include <float.h>
#include <set>
#include <assert.h>

#include "timing.h"
#include "Matrix.h"
#include "Array.h"
#include "utils.h"
#include "math.h"
#include "kernels.h"

using namespace std;

// external variables
double * kernel_args;
double * alphas;
double b;
int training_dataset_size;
int ndims;
int * y;
double **x;

// constants
double constant_c;
double constant_epsilon;
double constant_tau;
svm_kernel_type kernel_type;

// variables local to this file
double alpha_i_low_old;
double alpha_i_high_old;
double alpha_i_low_new;
double alpha_i_high_new;

bool train_model;
bool classify_dataset;

int amount_threads;

/**
 * returns dot product in kernel space between x[i0] and x[i1]
 * used to cache/study frequently used samples
 */
double kernel(int i0, int i1) {
	double * x0 = x[i0];
	double * x1 = x[i1];
	double result = kernel(x0, x1, ndims, kernel_type, kernel_args);
	return result;
}

/**
 * shows actual classifier result without being clipped to [-1;1]
 */
double classifierOutput(double * z) {
	double sum = 0;
	for (int i = 0; i < training_dataset_size; i++) {
		double alpha_i = alphas[i];
		if (alpha_i > 0) {
			double k_proj = kernel(z, x[i], ndims, kernel_type, kernel_args);
			sum += alpha_i * y[i] * k_proj;
		}
	}

	double out = sum + b;
	return out;
}

/**
 * shows actual classifier result without being clipped to [-1;1]
 * doesnt use bias
 */
double classifierOutputNoBias(Matrix<double> * model, double * z) {
	double sum = 0.0;
	for (int i = 0; i < model->nrows; i++) {
		double alpha = model->getElement(i, 0);
		double _y = model->getElement(i, 1);
		double * x_i = model->getElementPointer(i, 2);
		double k_proj = kernel(z, x_i, ndims, kernel_type, kernel_args);
		sum += alpha * _y * k_proj;
	}
	return sum;
}

void calculateBias(Matrix<double> * model) {
	double sum = 0.0;
//#pragma omp parallel for reduction (+: sum)
	for (int i = 0; i < model->nrows; i++) {
		double *x = model->getElementPointer(i, 2);
		double _y = (double) ((y[i]));
		sum += _y - classifierOutputNoBias(model, x);
	}
	b = sum / (double) model->nrows;
}

/**
 * shows classifier result, clipped to [-1;1]
 */
int classify(Matrix<double> * model, double * z) {
	double sum = 0.0;
	for (int i = 0; i < model->nrows; i++) {
		double alpha = model->getElement(i, 0);
		double _y = model->getElement(i, 1);
		double * x_i = model->getElementPointer(i, 2);
		double k_proj = kernel(z, x_i, ndims, kernel_type, kernel_args);
		sum += alpha * _y * k_proj;
	}

	double out = sum + b;
	if (out > 0.0)
		return 1;
	else
		return -1;
}

template<class T>
void processTestingDataset(Matrix<double> * model, T ** test_x, int * actual_y, int * predicted_y, int testing_dataset_size) {
	//confusion matrix
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;

	int errors = 0;
#pragma omp parallel for reduction(+:errors,tp,fp,tn,fn)
	for (int i = 0; i < testing_dataset_size; i++) {
		int predict = classify(model, test_x[i]);
		predicted_y[i] = predict;
		//confusion matrix
		if (predicted_y[i] == -1) {
			if (actual_y[i] == -1) {
				//TN
				tn++;
			} else {
				//FN
				fn++;
			}
		} else {
			if (actual_y[i] == -1) {
				//FP
				fp++;
			} else {
				//TP
				tp++;
			}
		}
		int class_err = actual_y[i] - predicted_y[i];
		if (class_err != 0)
			errors++;
	}
	cout << "Confusion matrix:" << endl;
	cout << "\t\t\tActual class" << endl;
	cout << "\t\t\t-1\t1" << endl;
	cout << "Predicted class\t-1\t" << tn << "\t" << fn << endl;
	cout << "\t\t1\t" << fp << "\t" << tp << endl;

	double precision = ((tp + fp) == 0 ? 0 : (double) (tp) / (double) (tp + fp));
	cout << "Precision: " << precision << endl;
	double recall = ((fn + tp) == 0 ? 0 : (double) (tp) / (double) (fn + tp));
	cout << "Recall: " << recall << endl;
	double false_positive_rate = ((fp + tn) == 0 ? 0 : (double) (fp) / (double) (fp + tn));
	cout << "False Positive Rate: " << false_positive_rate << endl;
	cout << "Specificity: " << 1.0 - false_positive_rate << endl;
	cout << "False Discovery Rate: " << ((fp + tp) == 0?0:(double) (fp) / (double) (fp + tp)) << endl;

	cout << "Accuracy: " << ((tp + tn + fp + fn) == 0?0:(double) (tp + tn) / (double) (tp + tn + fp + fn)) << endl;
	cout << "F-score: " << ((recall + precision) <FLT_MIN?0:(2.0 * recall * precision) / (recall + precision)) << endl;
	cout << "testing errors were " << errors << "/" << testing_dataset_size << " = " << (double) errors / (double) testing_dataset_size << endl;
}

void updateAlphasAdvanced(int &i_low, int &i_high, double &b_low, double &b_high) {
	// store old alphas for this iteration
	alpha_i_low_old = alphas[i_low];
	alpha_i_high_old = alphas[i_high];

	// targets
	double y_i_low = y[i_low];
	double y_i_high = y[i_high];

	// kernel computations
	double kxl_xl = kernel(i_low, i_low);
	double kxh_xh = kernel(i_high, i_high);
	double kxh_xl = kernel(i_high, i_low);

	// eta
	double eta = kxh_xh + kxl_xl - 2.0 * kxh_xl;

	double alphadiff = alpha_i_low_old - alpha_i_high_old;
	double sign = y_i_low * y_i_high;

	double alpha_l_upperbound, alpha_l_lowerbound;
	if (sign < 0.0) {
		if (alphadiff < 0) {
			alpha_l_lowerbound = 0;
			alpha_l_upperbound = constant_c + alphadiff;
		} else {
			alpha_l_lowerbound = alphadiff;
			alpha_l_upperbound = constant_c;
		}
	} else {
		double alpha_sum = alpha_i_low_old + alpha_i_high_old;
		if (alpha_sum < constant_c) {
			alpha_l_upperbound = alpha_sum;
			alpha_l_lowerbound = 0;
		} else {
			alpha_l_lowerbound = alpha_sum - constant_c;
			alpha_l_upperbound = constant_c;
		}
	}
	if (eta > 0) {
		alpha_i_low_new = alpha_i_low_old + y_i_low * (b_high - b_low) / eta;
		if (alpha_i_low_new < alpha_l_lowerbound) {
			alpha_i_low_new = alpha_l_lowerbound;
		} else
			if (alpha_i_low_new > alpha_l_upperbound) {
				alpha_i_low_new = alpha_l_upperbound;
			}
	} else {
		double slope = y_i_low * (b_high - b_low);
		double delta = slope * (alpha_l_upperbound - alpha_l_lowerbound);
		if (delta > 0) {
			if (slope > 0) {
				alpha_i_low_new = alpha_l_upperbound;
			} else {
				alpha_i_low_new = alpha_l_lowerbound;
			}
		} else {
			alpha_i_low_new = alpha_i_low_old;
		}
	}
	double alpha_l_diff = alpha_i_low_new - alpha_i_low_old;
	double alpha_h_diff = -sign * alpha_l_diff;
	alpha_i_high_new = alpha_i_high_old + alpha_h_diff;

	//store new alphas
	alphas[i_low] = alpha_i_low_new;
	alphas[i_high] = alpha_i_high_new;
}

void updateAlphas(int &i_low, int &i_high, double &b_low, double &b_high) {
	// store old alphas for this iteration
	alpha_i_low_old = alphas[i_low];
	alpha_i_high_old = alphas[i_high];

	// targets
	double y_i_low = y[i_low];
	double y_i_high = y[i_high];

	// kernel computations
	double kxl_xl = kernel(i_low, i_low);
	double kxh_xh = kernel(i_high, i_high);
	double kxh_xl = kernel(i_high, i_low);

	// eta
	double eta = kxh_xh + kxl_xl - 2.0 * kxh_xl;
	//ACHTUNG!!!!! eta can't be negative!!!
//	assert(eta > 0.0);
//	if (eta <= 0) {
//		cout << "eta:" << eta << " kxl_xl:" << kxl_xl << " kxh_xh:" << kxh_xh << " kxh_xl:" << kxh_xl << endl;
//	}

// compute new alphas
	alpha_i_low_new = alpha_i_low_old + y_i_low * (b_high - b_low) / eta;
	alpha_i_high_new = alpha_i_high_old + y_i_low * y_i_high * (alpha_i_low_old - alpha_i_low_new);

	// clip alphas in range 0 <= a <= C
	if (alpha_i_high_new < 0.0) {
		alpha_i_high_new = 0.0;
	}
	if (alpha_i_low_new < 0.0) {
		alpha_i_low_new = 0.0;
	}
	if (alpha_i_high_new > constant_c) {
		alpha_i_high_new = constant_c;
	}
	if (alpha_i_low_new > constant_c) {
		alpha_i_low_new = constant_c;
	}

	//store new alphas
	alphas[i_low] = alpha_i_low_new;
	alphas[i_high] = alpha_i_high_new;
}

void firstOrderHeuristicP(double & b_low, double & b_high, int & i_low, int & i_high, double * b_lows, double * b_highs, double * i_lows, double * i_highs,
		double * f) {
	//compute b_high, i_high, b_low, i_low
	//build I_high and I_low sets
	//also compute b_high & b_low using index sets I_high & I_low
	b_low = -(1E+37);
	b_high = (1E+37);

#pragma omp parallel
	{
		//initialize each thread's private vars
		int tid = omp_get_thread_num();

		double local_b_low, local_b_high; //local to each thread in firstOrderHeuristicP
		int local_i_low, local_i_high; //local to each thread in firstOrderHeuristicP
		local_b_low = -(1E+37);
		local_i_low = -1;
		local_b_high = (1E+37);
		local_i_high = -1;

		//each thread does a search on a subset
		int local_training_dataset_size = training_dataset_size / amount_threads;
		int i0 = (local_training_dataset_size * tid);
		int i1;
		if (tid == amount_threads - 1) {
			i1 = training_dataset_size - 1;
		} else {
			i1 = local_training_dataset_size * (tid + 1) - 1;
		}

		//do the search
		for (int i = i0; i < i1; i++) {
			double alpha_i = alphas[i];
			int y_i = y[i];
			// set belonging conditions
			bool I0 = (alpha_i > constant_epsilon) && (alpha_i < (constant_c - constant_epsilon));
			bool I1 = (y_i > 0) && fequal(alpha_i, 0, constant_epsilon);
			bool I2 = (y_i < 0) && fequal(alpha_i, constant_c, constant_epsilon);
			bool I3 = (y_i > 0) && fequal(alpha_i, constant_c, constant_epsilon);
			bool I4 = (y_i < 0) && fequal(alpha_i, 0, constant_epsilon);
			// check belonging to I_high
			if (I0 || I1 || I2) {
				double f_i = f[i];
				//compute b_high
				if (f_i < local_b_high) {
					local_b_high = f_i;
					//use first order heuristic to get next i_high
					local_i_high = i;
				}
			}

			// check belonging to I_low
			if (I0 || I3 || I4) {
				double f_i = f[i];
				//compute b_low
				if (f_i > local_b_low) {
					local_b_low = f_i;
					//use first order heuristic to get next i_low
					local_i_low = i;
				}
			}
		}

		//each thread stores the local result back to the array
		b_lows[tid] = local_b_low;
		b_highs[tid] = local_b_high;
		i_lows[tid] = local_i_low;
		i_highs[tid] = local_i_high;
	}

	//apply reduce/min|max to the above results
	b_low = b_lows[0];
	i_low = i_lows[0];
	b_high = b_highs[0];
	i_high = i_highs[0];
	for (int i = 1; i < amount_threads; i++) {
		if (b_lows[i] > b_low) {
			b_low = b_lows[i];
			i_low = i_lows[i];
		}
		if (b_highs[i] < b_high) {
			b_high = b_highs[i];
			i_high = i_highs[i];
		}
	}

}

void runSMO() {
	if (DEBUG)
		cout << "started SMO..." << endl;

//1. initialize
	b = 0.0;
	double constant_tau_doubled = constant_tau * 2.0;
//alpha_i=0 (and old alphas)
	double *f = new double[training_dataset_size];
	for (int i = 0; i < training_dataset_size; i++) {
		alphas[i] = 0.0;
	}
//f_i=-y_i;
	for (int i = 0; i < training_dataset_size; i++) {
		f[i] = -y[i];
	}
//2. initialize
	int iteration = 0;
//bhigh =  1,
	double b_high = -1.0;
//blow = 1,
	double b_low = 1.0;
//ihigh = min{i : yi = 1}
	int i_high = findMinimumPositionTarget_i(y, training_dataset_size, 1);
//ilow = max{i : yi =  -1}
	int i_low = findMinimumPositionTarget_i(y, training_dataset_size, -1);

//make sure everything is OK
	if (i_high < 0 || i_low < 0) {
		cout << "Err: couldn't initialize SMO's indices.." << endl;
		cout << "i_high:" << i_high << endl;
		cout << "i_low:" << i_low << endl;
		assert(i_high >= 0);
		assert(i_low >= 0);
	}
	// update alphas before entering loops
	updateAlphasAdvanced(i_low, i_high, b_low, b_high);

	// to be used in multi-threaded 1st order heuristic
	int h_storage_size = amount_threads;
	double *b_lows = new double[h_storage_size];
	double *b_highs = new double[h_storage_size];
	double *i_lows = new double[h_storage_size];
	double *i_highs = new double[h_storage_size];
	while (true) {
		if (DEBUG)
			if ((iteration & 256) && !(iteration & 128) && !(iteration & 64) && !(iteration & 32) && !(iteration & 16) && !(iteration & 8) && !(iteration & 4)
					&& !(iteration & 2) && !(iteration & 1)) {
				cout << "iteration:" << iteration << "\tgap:" << b_low - b_high << "\tb_low:" << b_low << "\tb_high:" << b_high << endl;
			}

		if (b_low <= b_high + constant_tau_doubled) {
			cout << "iteration:" << iteration << "\tgap:" << b_low - b_high << "\tb_low:" << b_low << "\tb_high:" << b_high << endl;
			break;
		}

		//check optimality conditions
		//update f_i for all i = 0...n-1
		double y_i_high = y[i_high];
		double y_i_low = y[i_low];
		double alpha_h_dif = alpha_i_high_new - alpha_i_high_old;
		double alpha_l_dif = alpha_i_low_new - alpha_i_low_old;

#pragma omp parallel for
		for (int i = 0; i < training_dataset_size; i++) {
			f[i] = f[i] + alpha_h_dif * y_i_high * kernel(i_high, i) + alpha_l_dif * y_i_low * kernel(i_low, i);
		}

		//compute b_high, i_high, b_low, i_low
		//using I_high and I_low sets
		firstOrderHeuristicP(b_low, b_high, i_low, i_high, b_lows, b_highs, i_lows, i_highs, f);

		//update the two lagrange multipliers
		updateAlphasAdvanced(i_low, i_high, b_low, b_high);
		iteration++;
	}

	if (DEBUG)
		cout << "converged!" << endl;
	cout << "total iterations: " << iteration << endl;

	cout << "computing bias..." << endl;
	//store bias
	//count amount of support vectors
	double sum = 0.0;
	int n_sv = 0;
#pragma omp parallel for reduction (+: sum, n_sv)
	for (int i = 0; i < training_dataset_size; i++) {
		double a_i = alphas[i];
		if (a_i > 0) {
			sum += (double) y[i] - classifierOutput(x[i]);
			n_sv++;
		}
	}
	b = sum / (double) n_sv;

	delete[] f;
	delete[] b_lows;
	delete[] b_highs;
	delete[] i_lows;
	delete[] i_highs;
}

#endif /* SMO_H_ */
