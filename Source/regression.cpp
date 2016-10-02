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


Regression::Regression(const DataSet& ds, const char* type):d_dset(ds)
{
    if(!strcmp(type, "Regression"))
    {
        d_reg_type = Regres;
        d_Theta.set_size((ds.N() + 1), 1);
    }
    else if(!strcmp(type, "Classification"))
    {
        d_reg_type = Classif;

        unsigned int features = ds.N();
        unsigned int classes = ds.K();

        if(!features)
        {
            cerr << "Regression: Regression class." << endl
                 << "Regression(const DataSet&, const char*) constructor" << endl
                 << "Feature size: " << features << " of the dataset cannot be 0." << endl;

            exit(1);
        }
        else if(!classes)
        {
            cerr << "Regression: Regression class." << endl
                 << "Regression(const DataSet&, const char*) constructor" << endl
                 << "Class size: " << classes << " of the dataset cannot be 0." << endl;

            exit(1);
        }

        d_Theta.set_size((features + 1), classes);
    }
    else
    {
        cerr << "Regression: Regression class." << endl
             << "Regression(const DataSet&, const char*) constructor" << endl
             << "Invalid regression type: " << type << endl;

        exit(1);
    }

    //--Initialize theta matrix with uniform random--//
    d_Theta.randu();

    //--Default 𝛼 and λ parameter values--//
    d_alpha = 0.1;
    d_lamda = 0.0;

    //--Initialize λ-graph file--//
    remove("../Output/lamda_cost.dat");
    d_lamdaCostGraph.open("../Output/lamda_cost.dat", ios_base::out);
    d_lamdaCostGraph << "#Lamda  #Cost" << endl;
}


Regression::~Regression()
{
    d_lamdaCostGraph.close();
}


double Regression::gradientdescent(mat X, const mat Y, const double delta, const unsigned int max_iter = 0)
{
    unsigned int m = X.n_rows;

    //--Adding bias terms to the data--//
    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    double c = 0;
    double c_prev=0;
    unsigned int it=0;

    //--Calculating pretrained cost of the dataset--//
    c = cost(X, Y);

    fstream costGraph;
    remove("../Output/cost.dat");
    costGraph.open("../Output/cost.dat", ios_base::out);
    costGraph << "#Iteration  #Cost" << endl;
    costGraph << it++ << " " << c << endl;

    cout << endl << "Training..." << endl;

    do
    {
        //--               𝛼   ∂J(Ө)  --//
        //-- Θ_j := Θ_j - --- ------- --//
        //--               m   ∂Θ_j   --//

        d_Theta = d_Theta - ((d_alpha/m) * derivative(X,Y));

        c_prev = c;
        c = cost(X, Y);

        costGraph << it++ << " " << c << endl;

    }while(fabs(c_prev - c) > delta && (max_iter ? ((it <= max_iter) ? true : false) : true));

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


string Regression::regressionType(void) const
{
    switch(d_reg_type)
    {
    case Regres:
    {
        return("Regression");
    }
    case Classif:
    {
        return("Classification");
    }
    default:
    {
        cerr << "Regression: Regression class." << endl
             << "string regressionType(void) method" << endl
             << "Invalid regression type: "<< d_reg_type  << endl;

        exit(1);
    }
    }
}


void Regression::set_regressionType(const string& reg_type)
{
    if(reg_type == "Regression")
    {
        d_reg_type = Regres;
    }
    else if(reg_type == "Classification")
    {
        d_reg_type = Classif;
    }
    else
    {
        cerr << "Regression: Regression class." << endl
             << "void set_regressionType(const string&) method" << endl
             << "Invalid regression type: "<< reg_type  << endl;

        exit(1);
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
