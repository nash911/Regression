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
    d_Theta.set_size((ds.N() + 1),1);
    d_Theta.randu();

    d_alpha = 0.1;
    d_lamda = 0.0;

    remove("../Output/lamda_cost.dat");
    d_lamdaCostGraph.open("../Output/lamda_cost.dat", ios_base::out);
    d_lamdaCostGraph << "#Lamda  #Cost" << endl;
}


Regression::~Regression()
{
    d_lamdaCostGraph.close();
}


double Regression::gradientdescent(const double delta)
{
    unsigned int m = d_dset.trainingSize();

    mat X = d_dset.XTrain();
    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    double c = cost(d_dset.XTrain(), d_dset.yTrain());
    double c_prev=0;
    unsigned int it=0;

    fstream costGraph;
    remove("../Output/cost.dat");
    costGraph.open("../Output/cost.dat", ios_base::out);
    costGraph << "#Iteration  #Cost" << endl;
    costGraph << it++ << " " << c << endl;

    cout << endl << "Training..." << endl;

    do
    {
        //--               𝛼   ∂h_Ө(X) --//
        //-- Θ_j := Θ_j - --- -------- --//
        //--               m    ∂Θ_j   --//

        //d_Theta = d_Theta - ((d_alpha/m) * ((X.t() * ((X * d_Theta) - y)) + (d_lamda * theta)));
        d_Theta = d_Theta - ((d_alpha/m) * derivative(X));

        c_prev = c;
        c = cost(d_dset.XTrain(), d_dset.yTrain());

        costGraph << it++ << " " << c << endl;

    }while(fabs(c_prev - c) > delta);

    cout << endl << "Finished training. Training details:"
         << endl << "Iterations: " << it
         << endl << "Delta_J(Theta): " << fabs(c_prev - c)
         << endl << "J(Theta): " << c << endl;

    costGraph.close();

    d_lamdaCostGraph << d_lamda << " " << c << endl;

    return c;
}


mat Regression::theta(void) const
{
    return d_Theta;
}


void Regression::printTheta(void) const
{
    cout << endl << "Theta:" << endl;
    for(unsigned int i=0; i<d_Theta.n_rows; i++)
    {
        for(unsigned int j=0; j<d_Theta.n_cols; j++)
        {
            cout << "theta[" << i << "," << j << "]:" << d_Theta(i,j) << endl;
        }
    }
}


void Regression::init_theta(void)
{
    if(d_Theta.n_rows == 0)
    {
        cerr << "Regression: Regression class." << endl
             << "void init_theta(void) method" << endl
             << "Cannot initialize empty theta matrix." << endl;

        exit(1);
    }
    else
    {
        d_Theta.randu();
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
