/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   L I N E A R   R E G R E S S I O N   C L A S S                                          */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "linear_regression.h"


LinearRegression::LinearRegression(const DataSet& ds):Regression(ds, "Regression")
{

}


vec LinearRegression::h_Theta(vec x) const
{
    if(x.n_rows != d_Theta.n_rows-1)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "vec h_Theta(vec) const method" << endl
             << "Size of vectors x: "<< x.n_rows  << " and Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    //--Insert 1.0 to the first row of the vector--//
    x.insert_rows(0,1);
    x(0) = 1.0;

    //--h_Ө(x) = Ө'x--//
    vec h = d_Theta.t() * x;

    return(h);
}


double LinearRegression::cost(mat& X, const mat& Y) const
{
    if(X.n_rows != Y.n_rows)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "double cost(mat&, const vec&) const method" << endl
             << "Rows of matrix X: "<< X.n_rows  << " must be equal to rows of Mx1 matrix Y: " << Y.n_rows << endl;

        exit(1);
    }

    if(X.n_cols != d_Theta.n_rows && X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "double cost(mat&, const vec&) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and row size of Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    double m = X.n_rows;
    vec cost;
    bool bias_term_added = false;

    if(X.n_cols == d_Theta.n_rows-1)
    {
        vec X_0 = ones<vec>(m);
        X.insert_cols(0, X_0);

        bias_term_added = true;
    }

    mat theta = d_Theta;
    theta.row(0).zeros();

    //--           _                                   _ --//
    //--        1 |  m                         n        |--//
    //--J(Ө) = ---|  ∑[h_Ө(x⁽i⁾) - y⁽i⁾]^2 +  λ∑(Ө_j)^2]|--//
    //--       2m |_ i                         j       _|--//

    mat residue = ((X * d_Theta) - Y);
    cost = (1.0/(2.0*m)) * ((residue.t() * residue) + (d_lamda * accu(theta % theta)));

    if(bias_term_added)
    {
        X.shed_col(0);
    }

    return(cost(0));
}


mat LinearRegression::derivative(const mat& X, const mat& Y) const
{
    //vec y = d_dset.yTrain();

    mat DeltaTheta;
    mat theta = d_Theta;
    theta.row(0).zeros();

    //-- ∂h_Ө(X)                         --//
    //-- -------- = (X'(XΘ - y)), ∀ j = 0--//
    //--   ∂Θ_j                          --//

    //-- ∂h_Ө(X)                                  --//
    //-- -------- = (X'(XΘ - y)) + λӨ_j), ∀ j >= 1--//
    //--   ∂Θ_j                                   --//

    DeltaTheta = (X.t() * ((X * d_Theta) - Y)) + (d_lamda * theta);

    return DeltaTheta;
}


vec LinearRegression::predict(mat X) const
{
    if(X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "vec predict(mat) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    unsigned int m = X.n_rows;

    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    return (X * d_Theta);
}


void LinearRegression::create_model(const unsigned int degree) const
{
    fstream model;
    remove("../Output/model.dat");
    model.open("../Output/model.dat", ios_base::out);
    model << "#Feature  #Target" << endl;

    double x = d_dset.Min()(0);
    double resolution = 1.0;

    unsigned int row_size = ((d_dset.Max()(0) - d_dset.Min()(0)) / resolution) + 1;
    mat X = zeros<mat>(row_size, 1);
    vec instance = zeros<vec>(1);

    vec prediction;
    double scaled_x;
    double scaled_x_1;

    for(unsigned int r=0; r<X.n_rows; r++)
    {
        scaled_x = (x-d_dset.Mean()(0))/d_dset.STDEV()(0);
        instance(0) = scaled_x;
        X(r,0) = scaled_x;

        x = x + resolution;
    }

    X = d_dset.mapFeatures(X, degree);
    prediction = predict(X);

    x = d_dset.Min()(0);
    for(unsigned int r=0; r<X.n_rows; r++)
    {
        model << x << " " << prediction(r) << endl;
        x = x + resolution;
    }

    model.close();
}
