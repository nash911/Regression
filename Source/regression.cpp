/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   R E G R E S S I O N   C L A S S                                                        */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "regression.h"


Regression::Regression(const DataSet& ds):d_dset(ds)
{
    d_Theta.set_size(ds.N() + 1);
    d_Theta.zeros();

    d_alpha = 0.1;
    d_lamda = 0.0;
}


vec Regression::theta(void) const
{
    return d_Theta;
}


void Regression::printTheta(void) const
{
    cout << endl << "Theta:" << endl;
    for(unsigned int i=0; i<d_Theta.n_rows; i++)
    {
        cout << "theta_" << i << ":" << d_Theta(i) << endl;
    }
}


void Regression::init_theta(void)
{
    if(d_Theta.n_rows == 0)
    {
        cerr << "Regression: Regression class." << endl
             << "void init_theta(void) method" << endl
             << "Cannot initialize empty theta vector." << endl;

        exit(1);
    }
    else
    {
        d_Theta.zeros();
    }
}


double Regression::alpha(void) const
{
    return d_alpha;
}


void Regression::set_alpha(const double alpha)
{
    if(alpha <= 0.0)
    {
        cerr << "Regression: Regression class." << endl
             << "void set_alpha(const double) method" << endl
             << "Alpha: "<< alpha  << " must be > 0." << endl;

        exit(1);
    }
    else
    {
        d_alpha = alpha;
    }
}


double Regression::lamda(void) const
{
    return d_lamda;
}


void Regression::set_lamda(const double lamda)
{
    if(lamda < 0.0)
    {
        cerr << "Regression: Regression class." << endl
             << "void set_lamda(const double) method" << endl
             << "Lamda: "<< lamda  << " must be >= 0." << endl;

        exit(1);
    }
    else
    {
        d_lamda = lamda;
    }
}
