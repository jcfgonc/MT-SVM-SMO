/*
 * kernels.h
 *
 *  Created on: Oct 28, 2011
 *  Author      : Joao Carlos jcfgonc@gmail.com
 *  License     : MIT License
 */

#ifndef KERNELS_H_
#define KERNELS_H_

enum svm_kernel_type {
	SVM_KT_LINEAR, SVM_KT_POLYNOMIAL, SVM_KT_RBF, SVM_KT_SIGMOID, SVM_KT_UKF
};

template<typename svm_kernel_type>
double kernel(double * x1, double * x2, int num_dimensions, svm_kernel_type kernel_type, double * kernel_args) {

	// select kernel_type from available_kernels
	switch (kernel_type) {
	case SVM_KT_LINEAR: {
		// 0 = linear kernel (default)
		// = x1.x2
		double sum = 0.0;
		for (int i = 0; i < num_dimensions; i++) {
			sum += x1[i] * x2[i];
		}
		return sum;
	}
	case SVM_KT_POLYNOMIAL: {
		//polynomial kernel
		//(a*(x1.x2)+b)^c
		// sum = x1.x2
		double sum = 0.0;
		for (int i = 0; i < num_dimensions; i++) {
			sum += x1[i] * x2[i];
		}
		double val = pow(kernel_args[0] * sum + kernel_args[1], kernel_args[2]);
		return val;
	}
	case SVM_KT_RBF: {
		// radial basis function (RBF) kernel, a = sigma
		// e^(-(1/(a^2)*(x1-x2)^2))
		double dif_squared = 0.0;
		for (int i = 0; i < num_dimensions; i++) {
			double x1_i = x1[i];
			double x2_i = x2[i];
			double _dif = x1_i - x2_i;
			double _dif_sq = _dif * _dif;
			dif_squared += _dif_sq;
		}
		double result = exp(-kernel_args[0] * dif_squared);
		return result;
	}
	case SVM_KT_SIGMOID: {
		// sigmoid kernel
		double sum = 0.0;
		for (int i = 0; i < num_dimensions; i++) {
			sum += x1[i] * x2[i];
		}
		double val = tanh(kernel_args[0] * sum + kernel_args[1]);
		return val;
	}
	case SVM_KT_UKF: {
		// universal kernel function
		// K(x1,x2) = a*(||x1-x2||^2+b^2)^-c
		double dif_squared = 0;
		for (int i = 0; i < num_dimensions; i++) {
			double _dif = x1[i] - x2[i];
			double _dif_sq = _dif * _dif;
			dif_squared += _dif_sq;
		}
		double result = kernel_args[0] * pow(dif_squared + kernel_args[1] * kernel_args[1], -kernel_args[2]);
		return result;
	}
	}
	return 0;
}

#endif /* KERNELS_H_ */
