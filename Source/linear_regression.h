/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   L I N E A R   R E G R E S S I O N   C L A S S   H E A D E R                            */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "regression.h"

class LinearRegression: public Regression
{
public:
    LinearRegression(const DataSet&);

    virtual double h_Theta(vec) const;
    virtual double cost(mat, const vec) const;
    virtual mat derivative(const mat&) const;

    vec predict(mat) const;
    //double test(mat, const vec) const;
    void create_model(const unsigned int) const;
};

#endif // LINEAR_REGRESSION_H
