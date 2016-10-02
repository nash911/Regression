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
    DataSet(const char*, const unsigned int, const double, const double, const bool);

    void extractMNISTData(const string);
    void extractDataFromFile(const char*, const unsigned int, const double, const double);

    int ReverseInt(int);
    void extractMNISTimg(const string, cube&);
    void extractMNISTlabel(const string, vec&);
    void oneHotEncode(const vec, mat&);
    void unrollCubetoMatrix(const cube&, mat&);

    unsigned int instanceSize(const char* const) const;
    unsigned int attributeSize(const char* const) const;

    void extractX(const char* const, const unsigned int, const unsigned int);
    void extractY(const char* const, const unsigned int, const unsigned int);    

    mat X() const;
    vec y() const;
    vec labels() const;

    unsigned int M() const;
    unsigned int N() const;
    unsigned int K() const;

    mat& XTrain();
    mat& yTrain();
    mat& Train_oneHotMatrix();
    unsigned int trainingSize(void) const;

    mat& XTest();
    mat& yTest();
    mat& Test_oneHotMatrix();
    unsigned int testSize(void) const;

    vec Mean() const;
    vec STDEV() const;
    vec Min() const;
    vec Max() const;
    vec normalizeFeatures(const vec);
    mat normalizeFeatures(const mat);
    void normalizeFeatures(cube&);

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

    cube d_train_img_cube;
    vec d_train_label_vec;
    mat d_train_1hot_mat;

    cube d_test_img_cube;
    vec d_test_label_vec;
    mat d_test_1hot_mat;

    vec d_class;

    vec d_mu;
    vec d_sigma;

    vec d_min;
    vec d_max;
};

#endif // DATASET_H
