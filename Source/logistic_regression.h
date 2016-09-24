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
    LogisticRegression(const DataSet&);

    virtual double h_Theta(vec) const;
    virtual double cost(mat, const vec) const;
    virtual mat derivative(const mat&) const;

    double classificationThreshold(void) const;
    void set_classificationThreshold(const double);

    vec sigmoid(const vec) const;
    uvec predict(mat) const;

    mat confusionMatrix(const mat, const vec) const;
    void print_confusionMatrix(const mat) const;
    double f1Score(const bool) const;
    double f1Score(const mat, const vec, const bool) const;

private:
    double d_classification_threshold;
};

#endif // LOGISTIC_REGRESSION_H
