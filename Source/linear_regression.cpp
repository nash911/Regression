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


LinearRegression::LinearRegression(const DataSet& ds):Regression(ds)
{

}


double LinearRegression::h_Theta(vec x) const
{
    if(x.n_rows != d_Theta.n_rows-1)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "double h_Theta(vec) const method" << endl
             << "Size of vectors x: "<< x.n_rows  << " and Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    //--Insert 1.0 to the first row of the vector--//
    x.insert_rows(0,1);
    x(0) = 1.0;

    //--h_Ө(x) = Ө'x--//
    vec h = d_Theta.t() * x;

    return(h(0));
}


double LinearRegression::cost(mat X, const vec y) const
{
    if(X.n_rows != y.n_rows)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "double cost(mat, const vec) const method" << endl
             << "Rows of matrix X: "<< X.n_rows  << " must be equal to rows of vector y: " << y.n_rows << endl;

        exit(1);
    }

    if(X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LinearRegression class." << endl
             << "double cost(mat, const vec) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    double m = X.n_rows;
    vec cost;

    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    mat theta = d_Theta;
    theta(0,0) = 0;

    //--           _                                   _ --//
    //--        1 |  m                         n        |--//
    //--J(Ө) = ---|  ∑[h_Ө(x⁽i⁾) - y⁽i⁾]^2 +  λ∑(Ө_j)^2]|--//
    //--       2m |_ i                         j       _|--//

    cost = (1.0/(2.0*m)) * ((((X * d_Theta) - y).t() * ((X * d_Theta) - y)) + (d_lamda * accu(theta % theta)));

    return(cost(0));
}


double LinearRegression::gradientdescent(const double delta)
{
    unsigned int m = d_dset.trainingSize();

    mat X = d_dset.XTrain();
    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    vec y = d_dset.yTrain();

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
        mat theta = d_Theta;
        theta(0,0) = 0;

        //--                1                               --//
        //-- Θ_j := Θ_j - 𝛼---(X'(XΘ - y))           ∀ j = 0--//
        //--                m                               --//

        //--                1                               --//
        //-- Θ_j := Θ_j - 𝛼---((X'(XΘ - y)) + λӨ_j) ∀ j >= 1--//
        //--                m                               --//

        d_Theta = d_Theta - ((d_alpha/m) * ((X.t() * ((X * d_Theta) - y)) + (d_lamda * theta)));

        c_prev = c;
        c = cost(d_dset.XTrain(), d_dset.yTrain());

        costGraph << it++ << " " << c << endl;

    }while(fabs(c_prev - c) > delta);

    cout << endl << "Finished training. Training details:"
         << endl << "Iterations: " << it
         << endl << "Delta_J(Theta): " << fabs(c_prev - c)
         << endl << "J(Theta): " << c
         << endl << "Theta: " << d_Theta << endl;

    costGraph.close();

    d_lamdaCostGraph << d_lamda << " " << c << endl;

    return c;
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
