//============================================================================
// Name        : svm_smo.cpp
// Author      : Joao Carlos jcfgonc@gmail.com
// License     : MIT License
//============================================================================

typedef unsigned int uint;

//value separator tag for the csv files
#define VALUE_SEPARATOR ","

#define BUFFERING_BLOCK_SIZE (1<<24)

#define DEBUG 0

#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <float.h>
using namespace std;

#include "Settings.h"
#include "timing.h"
#include "Matrix.h"
#include "Array.h"
#include "utils.h"
#include "smo.h"

extern double constant_c;
extern double constant_epsilon;
extern double constant_tau;

extern bool train_model;
extern bool classify_dataset;

extern int amount_threads;
extern double * alphas;

//indices are returned in alpha_indices, total count is returned
int getSupportVectorIndices(double * alphas, int * alpha_indices, int size) {
	int non_zero_counter = 0;
	for (int i = 0; i < size; i++) {
		double alpha = alphas[i];
		if (alpha > 0.0) {
			alpha_indices[non_zero_counter] = i;
			non_zero_counter++;
		}
	}
	return non_zero_counter;
}

void saveModel(char * model_filename, Matrix<double> *model) {
	if (DEBUG)
		cout << "saving model to file..." << endl;
	char * WRITE_BUF = new char[BUFFERING_BLOCK_SIZE];
	FILE *model_file;
	model_file = fopen(model_filename, "w");
	if (model_file) {
		//buffer process
		setvbuf(model_file, WRITE_BUF, _IOFBF, BUFFERING_BLOCK_SIZE);
		//first line gives the amount of support vectors
		fprintf(model_file, "%d\n", model->nrows);
		//second line gives the amount of features
		fprintf(model_file, "%d\n", model->ncols - 2);
		//the remaining lines are in the form:
		//alpha_i | class_i | attribute_0 ... attribute_n-1
		for (int sv_i = 0; sv_i < model->nrows; sv_i++) {
			for (int col = 0; col < model->ncols; col++) {
				fprintf(model_file, "%f", model->getElement(sv_i, col));
				if (col < model->ncols - 1)
					fprintf(model_file, ",");
			}
			fprintf(model_file, "\n");
		}
		fclose(model_file);
	} else
		cout << "Err: Unable to open model file for write." << endl;
	delete[] WRITE_BUF;
}

// Dr. Floyd - CODE AS FOLLOWS...
int main(int argc, char **argv) {
	//disable stdout buffering
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);

	cout << "Multi-Threaded SVM using the SMO algorithm" << endl;
	cout << "(C) 2012 Joao Goncalves: jcfgonc@gmail.com" << endl;

	char * training_filename = NULL;
	char * testing_filename = NULL;
	char * model_filename = NULL;
	char * classification_results_filename = NULL;
	double time_program_start, time_program_end;

	kernel_type = SVM_KT_LINEAR;
	kernel_args = new double[4];
	kernel_args[0] = 1.0;
	kernel_args[1] = 1.0;
	kernel_args[2] = 1.0;
	kernel_args[3] = 1.0;

	constant_c = 1.0;
	constant_epsilon = 0.00001;
	constant_tau = 0.001;
	int positive_class = 1;

	amount_threads = 0;

	bool arguments_error = false;

	//read arguments and compile them
	Settings settings(argc, argv);
	unsigned int aa = settings.getAmountArguments();
	//go through all arguments
	for (unsigned int i = 0; i < aa; i++) {
		Argument* a = settings.getArgument(i);
		//training file
		if (strcmp(a->argument, "-trainingset") == 0) {
			//cout << "extracting training file" << endl;
			if (a->value != NULL) {
				training_filename = a->value;
			} else {
				cout << "no training file was given" << endl;
				arguments_error = true;
			}
		} else
			//classifying file
			if (strcmp(a->argument, "-testingset") == 0) {
				//cout << "extracting testing file" << endl;
				if (a->value != NULL) {
					testing_filename = a->value;
				} else {
					cout << "no testing file was given" << endl;
					arguments_error = true;
				}
			} else
				//classification results file
				if (strcmp(a->argument, "-cr") == 0) {
					//cout << "extracting classification results file" << endl;
					if (a->value != NULL) {
						classification_results_filename = a->value;
					} else {
						cout << "no classification results file was given" << endl;
						arguments_error = true;
					}
				} else
					//model file
					if (strcmp(a->argument, "-model") == 0) {
						//cout << "extracting model file" << endl;
						if (a->value != NULL) {
							model_filename = a->value;
						} else {
							cout << "no model file given" << endl;
							arguments_error = true;
						}
					} else
						//train?
						if (strcmp(a->argument, "-train") == 0) {
							//cout << "user wants to train model" << endl;
							train_model = true;
						} else
							//classify?
							if (strcmp(a->argument, "-classify") == 0) {
								//cout << "user wants to classify dataset" << endl;
								classify_dataset = true;
							} else
								//kernel type
								if (strcmp(a->argument, "-k") == 0) {
									//cout << "extracting kernel type" << endl;
									if (a->value != NULL) {
										if (strcmp(a->value, "lin") == 0) {
											kernel_type = SVM_KT_LINEAR;
										} else
											if (strcmp(a->value, "pol") == 0) {
												kernel_type = SVM_KT_POLYNOMIAL;
											} else
												if (strcmp(a->value, "rbf") == 0) {
													kernel_type = SVM_KT_RBF;
												} else
													if (strcmp(a->value, "sig") == 0) {
														kernel_type = SVM_KT_SIGMOID;
													} else
														if (strcmp(a->value, "ukf") == 0) {
															kernel_type = SVM_KT_UKF;
														} else {
															cout << "unknown kernel type: " << a->value << endl;
															arguments_error = true;
														}
									} else {
										cout << "no kernel type was given" << endl;
										arguments_error = true;
									}
								} else
									//kernel arguments
									//a
									if (strcmp(a->argument, "-a") == 0) {
										//cout << "extracting argument <a>" << endl;
										if (a->value != NULL) {
											kernel_args[0] = atof(a->value);
										} else {
											cout << "no argument <a> was given" << endl;
											arguments_error = true;
										}
									} else
										//b
										if (strcmp(a->argument, "-b") == 0) {
											//cout << "extracting argument <b>" << endl;
											if (a->value != NULL) {
												kernel_args[1] = atof(a->value);
											} else {
												cout << "no argument <b> was given" << endl;
												arguments_error = true;
											}
										} else
											//c
											if (strcmp(a->argument, "-c") == 0) {
												//cout << "extracting argument <c>" << endl;
												if (a->value != NULL) {
													kernel_args[2] = atof(a->value);
												} else {
													cout << "no argument <c> was given" << endl;
													arguments_error = true;
												}
											} else
												//penalization constant
												if (strcmp(a->argument, "-C") == 0) {
													//cout << "extracting penalization constant C" << endl;
													if (a->value != NULL) {
														constant_c = atof(a->value);
													} else {
														cout << "no penalization constant was given" << endl;
														arguments_error = true;
													}
												} else
													//optimality conditions tolerance
													if (strcmp(a->argument, "-eps") == 0) {
														//cout << "extracting optimality conditions tolerance" << endl;
														if (a->value != NULL) {
															constant_epsilon = atof(a->value);
														} else {
															cout << "no optimality conditions tolerance was given" << endl;
															arguments_error = true;
														}
													} else
														//optimality gap size
														if (strcmp(a->argument, "-tau") == 0) {
															//cout << "extracting optimality gap size" << endl;
															if (a->value != NULL) {
																constant_tau = atof(a->value);
															} else {
																cout << "no optimality gap size was given" << endl;
																arguments_error = true;
															}
														} else
															//amount of threads gap size
															if (strcmp(a->argument, "-threads") == 0) {
																//cout << "extracting amount of threads" << endl;
																if (a->value != NULL) {
																	amount_threads = atoi(a->value);
																} else {
																	cout << "no amount of threads was given" << endl;
																	arguments_error = true;
																}
															} else
																//positive class
																if (strcmp(a->argument, "-positive") == 0) {
																	//cout << "extracting amount of threads" << endl;
																	if (a->value != NULL) {
																		positive_class = atoi(a->value);
																	} else {
																		cout << "no positive class id was given" << endl;
																		arguments_error = true;
																	}
																}

	}

	//for training we require the training dataset... duh
	if (train_model) {
		if (training_filename == NULL) {
			cout << "Error: no training dataset was given - Aborting." << endl;
			arguments_error = true;
		} else {
			//cout << "training dataset is " << training_filename << endl;
		}
	}

	//for classifying we require both the training and testing datasets
	if (classify_dataset) {
		//if in this execution the model is not trained, it must be read from somewhere...
		if (train_model == false && model_filename == NULL) {
			cout << "Error: no model file was given." << endl;
			return -1;
		} else {
			//cout << "model file is " << model_filename << endl;
		}
		if (testing_filename == NULL) {
			cout << "Error: no testing dataset was given." << endl;
			return -1;
		} else {
			//cout << "testing dataset is " << model_filename << endl;
		}
	}

	if (classify_dataset == false && train_model == false) {
		cout << "Error: the program was not instructed to train nor to classify." << endl;
		arguments_error = true;
	}

	if (arguments_error) {
		cout << "Error: invalid arguments." << endl;
		cout << "----------------------------------------------------------" << endl;
		cout << "The arguments are the following:" << endl;
		cout << "" << endl;

		cout << "to train using the training samples" << endl;
		cout << "\t -train" << endl;
		cout << "" << endl;

		cout << "to classify using the trained svm model" << endl;
		cout << "\t -classify" << endl;
		cout << "" << endl;

		cout << "file with the training set (filename) - required:" << endl;
		cout << "\t -trainingset <training file>" << endl;
		cout << "" << endl;

		cout << "file with the testing set (filename) - required:" << endl;
		cout << "\t -testingset <training file>" << endl;
		cout << "" << endl;

		cout << "file where to store the trained svm model (filename):" << endl;
		cout << "\t -model <output file>" << endl;
		cout << "" << endl;

		cout << "which kernel to use (text):" << endl;
		cout << "\t -k <type>" << endl;
		cout << "\t where <type> can be one of the following:" << endl;
		cout << "\t\t lin - for the linear kernel: K(x1,x2) = x1.x2" << endl;
		cout << "\t\t pol - for the polynomial kernel: K(x1,x2) = a*(x1.x2+b)^c" << endl;
		cout << "\t\t rbf - for the gaussian kernel: K(x1,x2) = e^(-a*||x1-x2||^2)" << endl;
		cout << "\t\t sig - for the sigmoid kernel: K(x1,x2) = tanh(a*(x1.x2)+b)" << endl;
		cout << "\t\t ukf - for the universal function kernel: K(x1,x2) = a*(||x1-x2||^2+b^2)^-c" << endl;
		cout << "\t being x1.x2 the dot product between vectors x1 and x2" << endl;
		cout << "" << endl;

		cout << "kernel arguments (decimal number):" << endl;
		cout << "\t -a <value>" << endl;
		cout << "\t -b <value>" << endl;
		cout << "\t -c <value>" << endl;
		cout << "" << endl;

		cout << "penalization constant C (decimal number):" << endl;
		cout << "\t -C <value>" << endl;
		cout << "" << endl;

		cout << "optimality conditions tolerance, Epsilon, which allows some numerical uncertainty on the heuristics (decimal number):" << endl;
		cout << "\t -eps <value>" << endl;
		cout << "" << endl;

		cout << "optimality gap size, Tau, which regulates the training convergence (decimal number):" << endl;
		cout << "\t -tau <value>" << endl;
		cout << "" << endl;

		cout << "amount of threads to use in trainer and classifier (integer, 0 = automatic):" << endl;
		cout << "\t -threads <value>" << endl;
		cout << "" << endl;

		cout << "ABORTING." << endl;
		return -1;
	}

	switch (kernel_type) {
	case SVM_KT_RBF:
		if (DEBUG)
			cout << "using RBF kernel with gamma = " << kernel_args[0] << endl;
		break;
	case SVM_KT_LINEAR:
		if (DEBUG)
			cout << "using linear kernel" << endl;
		break;
	case SVM_KT_POLYNOMIAL:
		if (DEBUG)
			cout << "using polynomial kernel" << endl;
		break;
	case SVM_KT_SIGMOID:
		if (DEBUG)
			cout << "using sigmoid kernel" << endl;
		break;
	case SVM_KT_UKF:
		if (DEBUG)
			cout << "using universal kernel function with L = " << kernel_args[0] << " b (sigma) = " << kernel_args[1] << " and c (alpha) = " << kernel_args[2]
					<< endl;
		break;
	}

	if (constant_c <= 0) {
		cout << "Error: invalid value for C" << endl;
		return -1;
	}
	if (DEBUG)
		cout << "C = " << constant_c << endl;
	if (DEBUG)
		cout << "epsilon = " << constant_epsilon << endl;
	if (constant_tau <= 0) {
		cout << "Error: invalid value for epsilon" << endl;
		return -1;
	}
	if (DEBUG)
		cout << "tau = " << constant_tau << endl;

	// read training dataset file
	// read training dataset file
	// read training dataset file

	if (amount_threads > 0)
		omp_set_num_threads(amount_threads);
	else {
		amount_threads = omp_get_num_procs();
	}
	cout << "using " << omp_get_max_threads() << " threads" << endl;

	//build matrix for holding training data set
	//cout << "reading training dataset file:" << training_filename << endl;
	FILE *f_input = fopen(training_filename, "r");
	if (f_input == NULL) {
		cout << "error while reading training dataset file" << endl;
		return -1;
	}
	ndims = getNumberOfColumns(f_input) - 1;
	training_dataset_size = getNumberOfSamples(f_input);
	//cout << "allocating storage for training dataset:" << training_filename << endl;
	Matrix<double> * training_features = new Matrix<double>(training_dataset_size, ndims);
	Array<int> * training_classes = new Array<int>(training_dataset_size);
	readDataSet(f_input, training_features, training_classes, ndims + 1, positive_class);
	fclose(f_input);

	y = training_classes->array;
	x = training_features->matrix;

	alphas = new double[training_dataset_size];
	int n_sv;
	Matrix<double> * model;
	//train model if requested
	if (train_model) {
		shrDeltaT(1);
		shrDeltaT(1);
		time_program_start = shrDeltaT(1);
		runSMO();

		//compress alphas array
		if (DEBUG)
			cout << "creating model..." << endl;
		int * alpha_indices = new int[training_dataset_size];
		n_sv = getSupportVectorIndices(alphas, alpha_indices, training_dataset_size);

		//create a matrix to hold the model
		//structure: alpha_i | class_i | attribute_0 ... attribute_n-1
		model = new Matrix<double>(n_sv, ndims + 2);
#pragma omp parallel for
		for (int row = 0; row < n_sv; row++) {
			//the index of current non zero alpha (support vector) on the original dataset
			int index = alpha_indices[row];
			//the value of alpha (lagrange multiplier)
			double alpha_i = alphas[index];
			//set alpha on model
			model->setElement(alpha_i, row, 0);
			//the class associated with current alpha
			int c_i = training_classes->getElement(index);
			//set class on model
			model->setElement(c_i, row, 1);
			//set the remaining elements as the features
			for (int feature_i = 0; feature_i < training_features->ncols; feature_i++) {
				//get the original attribute
				double attribute = training_features->getElement(index, feature_i);
				//copy to the model
				model->setElement(attribute, row, feature_i + 2);
			}
		}
		delete[] alpha_indices;
//		cout << "calculating bias..." << endl;
//		calculateBias(model);
		cout << "bias: " << b << " nSVs: " << n_sv << endl;

		time_program_end = shrDeltaT(1);
		printf("training took %f s\n", time_program_end - time_program_start);

		//if requested save model to a file
		if (model_filename != NULL) {
			saveModel(model_filename, model);
		}
	}

	if (classify_dataset) {
		// if in this call the model hasn't been created, load it
		if (!train_model) {
			//TODO: check this
			cout << "loading model from file..." << endl;
			ifstream model_file(model_filename);
			if (model_file.is_open()) {
				//first line tells the amount of SVs
				model_file >> n_sv;
				//second tells the amount of features
				model_file >> ndims;
				//create the model
				model = new Matrix<double>(n_sv, ndims + 2);
				for (int row = 0; row < model->nrows; row++) {
					for (int col = 0; col < model->ncols; col++) {
						double val;
						model_file >> val;
						model->setElement(val, row, col);
					}
				}
				model_file.close();
			} else {
				cout << "Err: Unable to open model file for reading." << endl;
				return -1;
			}
		}

		// read testing dataset file
		// read testing dataset file
		// read testing dataset file

		//build matrix for holding testing data set
		FILE *f_input = fopen(testing_filename, "r");
		if (f_input == NULL) {
			cout << "error while reading testing dataset file" << endl;
			return -1;
		}
		int testing_dataset_size = getNumberOfSamples(f_input);
		Matrix<double> * testing_features = new Matrix<double>(testing_dataset_size, ndims);
		Array<int> * testing_classes = new Array<int>(testing_dataset_size);
		readDataSet<double>(f_input, testing_features, testing_classes, ndims + 1, positive_class);
		fclose(f_input);

		Array<int> * testing_results = new Array<int>(testing_dataset_size);

		shrDeltaT(1);
		shrDeltaT(1);
		time_program_start = shrDeltaT(1);
		processTestingDataset<double>(model, testing_features->matrix, testing_classes->array, testing_results->array, testing_dataset_size);
		time_program_end = shrDeltaT(1);
		printf("classification took %f s\n", time_program_end - time_program_start);

		//if requested save results to a file (just the classifications)
		if (classification_results_filename != NULL) {
			if (DEBUG)
				cout << "saving classification results to file..." << endl;

			char * wbuf = new char[BUFFERING_BLOCK_SIZE];
			FILE *model_file;
			model_file = fopen(classification_results_filename, "w");
			if (model_file) {
				//buffer process
				setvbuf(model_file, wbuf, _IOFBF, BUFFERING_BLOCK_SIZE);
				//first line tells the amount of samples
				fprintf(model_file, "%d\n", testing_dataset_size);
				for (int i = 0; i < testing_dataset_size; i++) {
					fprintf(model_file, "%d\n", testing_results->getElement(i));
				}
				fclose(model_file);
			} else
				cout << "Err: Unable to open classification results file for write." << endl;
			delete[] wbuf;
		}

		delete testing_features;
		delete testing_classes;
		delete testing_results;
	}

	delete alphas;
	delete training_features;
	delete training_classes;
	if (DEBUG)
		cout << "exiting..." << endl;
	return 0;
}
