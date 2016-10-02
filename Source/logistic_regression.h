/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   L O G I S T I C   R E G R E S S I O N   C L A S S   H E A D E R                        */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "regression.h"

#define CLASSIFICATION_THRESHOLD 0.5

class LogisticRegression: public Regression
{
public:
    enum ClassificationFunction{Sigmoid, Softmax};

    LogisticRegression(const DataSet&);

    virtual vec h_Theta(vec) const;
    virtual double cost(mat&, const mat&) const;
    virtual mat derivative(const mat&, const mat&) const;

    string classificationFunction(void) const;
    void set_classificationFunction(const string&);

    double classificationThreshold(void) const;
    void set_classificationThreshold(const double);

    mat sigmoid(const mat) const;
    mat softmax(const mat) const;
    mat predict(mat, const mat) const;

    umat confusionMatrix(const mat, const mat) const;
    void print_confusionMatrix(const umat) const;
    double f1Score(const mat, const mat, const bool) const;

private:
    ClassificationFunction d_class_func;
    double d_classification_threshold;
};

#endif // LOGISTIC_REGRESSION_H
