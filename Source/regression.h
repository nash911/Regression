/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   R E G R E S S I O N   C L A S S   H E A D E R                                          */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#ifndef REGRESSION_H
#define REGRESSION_H

#include<iostream>
#include<fstream>
#include<math.h>

#include "armadillo"
#include "dataset.h"

using namespace std;
using namespace arma;

class Regression
{
public:
    enum RegressionType{Regres, Classif};

    Regression(const DataSet&, const char*);
    ~Regression();

    double gradientdescent(mat, const mat, const double, const unsigned int);

    mat theta(void) const;
    void init_theta(void);
    void printTheta(void) const;

    string regressionType(void) const;
    void set_regressionType(const string&);

    double alpha(void) const;
    void set_alpha(const double);

    double lamda(void) const;
    void set_lamda(const double);

    virtual vec h_Theta(vec) const = 0;
    virtual double cost(mat&, const mat&) const = 0;
    virtual mat derivative(const mat&, const mat&) const = 0;

protected:
    mat d_Theta;

    const DataSet& d_dset;
    RegressionType d_reg_type;

    double d_alpha;
    double d_lamda;

    fstream d_lamdaCostGraph;
};

#endif // REGRESSION_H
