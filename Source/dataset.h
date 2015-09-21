/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   D A T A   S E T   C L A S S   H E A D E R                                              */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#ifndef DATASET_H
#define DATASET_H

#include<iostream>
#include<fstream>
#include<math.h>

#include "armadillo"

using namespace std;
using namespace arma;

class DataSet
{
public:
    DataSet(const char*, const unsigned int, const double, const double);

    unsigned int instanceSize(const char* const) const;
    unsigned int attributeSize(const char* const) const;

    void extractX(const char* const, const unsigned int, const unsigned int);
    void extractY(const char* const, const unsigned int, const unsigned int);    

    mat X() const;
    vec y() const;
    unsigned int M() const;
    unsigned int N() const;

    mat XTrain() const;
    vec yTrain() const;
    unsigned int trainingSize(void) const;

    mat XTest() const;
    vec yTest() const;
    unsigned int testSize(void) const;

    vec Mean() const;
    vec STDEV() const;
    vec normalizeFeatures(const vec);
    mat normalizeFeatures(const mat);

    mat exponents(const mat, const unsigned int) const;
    mat mapFeatures(const mat, const unsigned int) const;

    void segmentDataSet(const double, const double);

    void printDataSet() const;
    void printTrainingSet() const;
    void printTestSet() const;

    void saveToFile(const mat) const;

private:
    mat d_X;
    vec d_y;

    mat d_X_train;
    vec d_y_train;

    mat d_X_test;
    vec d_y_test;

    vec mu;
    vec sigma;
};

#endif // DATASET_H
