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
    Regression(const DataSet&);
    ~Regression();

    vec theta(void) const;
    void init_theta(void);
    void printTheta(void) const;

    double alpha(void) const;
    void set_alpha(const double);

    double lamda(void) const;
    void set_lamda(const double);

    virtual double h_Theta(vec) const = 0;
    virtual double cost(mat, const vec) const = 0;
    virtual double gradientdescent(const double) = 0;

protected:
    vec d_Theta;

    const DataSet& d_dset;

    double d_alpha;
    double d_lamda;

    fstream d_lamdaCostGraph;
};

#endif // REGRESSION_H
