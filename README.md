# Introduction

MT-SVM-SMO is a prototype of a Support Vector Machine that served as a playground for a later GPGPU implementation 
written in C++/CUDA. The main features of this implementation is the support for multi-threaded training/classification 
 and the support for five types of kernel functions:

* linear
* polynomial
* radial basis function (RBF)
* sigmoid
* universal kernel function (UKF)

This prototype produced a paper that was published in the ICONIP 2012 Neural Information Processing conference.
You can find it either in the 'paper' folder as a rough draft or the final published version at 
[Springer](https://link.springer.com/chapter/10.1007/978-3-642-34481-7_75).

You can find its abstract below:

> Support Vector Machines (SVM) have become indispensable tools in the area of pattern recognition. They show 
powerful classification and regression performance in highly non-linear problems by mapping the input vectors 
nonlinearly into a high-dimensional feature space through a kernel function. However, the optimization task is 
numerically expensive since single-threaded implementations are hardly able to cope up with the complex learning 
task. In  this paper, we present a multi-threaded implementation of the Sequential Minimum Optimization (SMO) 
which reduces the numerical complexity by parallelizing the KKT conditions update, the calculus of the hyperplane 
offset and the classification task. Our preliminary performance results in a few benchmark datasets and in a MP3 
steganalysis problem are competitive compared to state-of-the-art tools while the execution running times were 
considerably faster. 

# Licensing

mapperMO is released under the MIT License, a copy of which is included in this directory.

# People

The primary contributors to the MT-SVM-SMO are:

* João Gonçalves
* Noel Lopes
* Bernardete Ribeiro

Please email jcfgonc@gmail.com with questions, comments, and bug reports.
