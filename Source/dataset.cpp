/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   D A T A   S E T   C L A S S                                                            */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "dataset.h"

// CONSTRUCTOR

/// Creates a Data Set object.
/// Extracts training data containing features and target from the file whose path and name is passed as a parameter.
/// Creates new polynomial features through feature mapping.
/// Calculates the mean and standard deviation of the data features and then the features are Mean Normalized.
/// Shuffels the data set and divides it into training and test sets.
/// @param fileName Path and name of the file containing the training data.
/// @param degree Specifies the degree of polynomial for feature mapping. degree ≥ 1. degree = 1 ensures data set remains unchanged.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.

DataSet::DataSet(const char* fileName, const unsigned int degree=1, const double trainPercent=70, const double testPercent=30)
{
    if(!fileName)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double) constructor." << endl
             << "Cannot open data file: " << fileName
             << endl;

        exit(1);
    }

    if(degree == 0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double) constructor." << endl
             << "Parameter degree: " << degree << " for polynomial feature mapping has to be >= 1 "
             << endl;

        exit(1);
    }

    if(trainPercent <= 0.0 || testPercent < 0.0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double) constructor." << endl
             << "Training set = " << trainPercent << "% has to be > 0% and Test set = "<< testPercent << "% has to be >= 0%."
             << endl;

        exit(1);
    }

    if(trainPercent + testPercent != 100.0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double) constructor." << endl
             << "Training set = " << trainPercent << " + Test set = "<< testPercent << " has to be equal to 100%."
             << endl;

        exit(1);
    }

    //--Extract no. of instances and attributes of the data set on file--//
    unsigned int instSize = instanceSize(fileName);
    unsigned int attSize = attributeSize(fileName);

    //--Extract feature set from data file--//
    d_X.set_size(instSize, attSize);
    d_X.zeros();
    extractX(fileName, instSize, attSize);

    //--Extract targets from data file--//
    d_y.set_size(instSize);
    d_y.zeros();
    extractY(fileName, instSize, attSize);

    if(d_X.n_rows != d_y.n_rows)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double) constructor." << endl
             << "No. of instances in matrix X: " << d_X.n_rows << "  and vector y: " << d_y.n_rows << " do not match."
             << endl;

        exit(0);
    }

    //--Create new features through Feature Mapping--//
    //d_X = mapFeatures(d_X, degree);

    //--Calculate and store the μ and σ of the features--//
    //mu = mean(featMap_X).t();
    //sigma = stddev(featMap_X).t();
    d_mu = mean(d_X).t();
    d_sigma = stddev(d_X).t();

    //d_mu = zeros<vec>(N());
    //d_sigma = ones<vec>(N());

    //--Calculate and store the min and max of the features--//
    d_min = min(d_X).t();
    d_max = max(d_X).t();

    //--Normalize features--//
    //mat norm_X = normalizeFeatures(featMap_X);
    d_X = normalizeFeatures(d_X);

    //--Create new features through Feature Mapping--//
    d_X = mapFeatures(d_X, degree);

    //--Shuffle the data and segment into training and test sets--//
    //segmentDataSet(norm_X, trainPercent, testPercent);
    segmentDataSet(trainPercent, testPercent);
}


// unsigned int instanceSize(const char* const) const method

/// Extracts and returns the number of instances (m) on the data file.
/// @param fileName Path and name of the file containing the training data.

unsigned int DataSet::instanceSize(const char* const fileName) const
{
    double numInstances = 0;

    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "unsigned int instanceSize(const char* const) const method" << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    string line;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting number of instances on file--//
    while(!inputFile.eof())
    {
        getline(inputFile, line);
        numInstances++;
    }

    cout << endl << "Number of instances on file: " << numInstances << endl;
    inputFile.close();

    return numInstances;
}


// unsigned int attributeSize(const char* const) const method

/// Extracts and returns the number of features (n) on the data file.
/// @param fileName Path and name of the file containing the training data.

unsigned int DataSet::attributeSize(const char* const fileName) const
{
    double numAttributes = 0;

    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "unsigned int attributeSize(const char* const) const method" << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    string line;
    string attribute;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    stringstream ssLine(line);
    ssLine >> attribute;

    //--Extracting the number of attributes on file--//
    while(ssLine >> attribute)
    {
        numAttributes++;
    }


    cout << endl << "Number of attributes on file: " << numAttributes << endl;
    inputFile.close();

    return numAttributes;
}


// void extractX(const char* const, const unsigned int, const unsigned int) method

/// Extracts attributes of the data set from the file.
/// @param fileName Path and name of the file containing the training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void DataSet::extractX(const char* const fileName, const unsigned int M, const unsigned int N)
{
    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "void extractX(const char* const, const unsigned int, const unsigned int) method" << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }


    string line;
    double X_i;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting features from file, one instance at a time--//
    for(unsigned int m=0; m<M; m++)
    {
        stringstream ssLine(line);
        for(unsigned int n=0; n<N; n++)
        {
            ssLine >> X_i;
            d_X(m,n) = X_i;
        }
        getline(inputFile, line);
    }

}


// void extractY(const char* const, const unsigned int, const unsigned int) method

/// Extracts targest of the data set from the file.
/// @param fileName Path and name of the file containing the training data.
/// @param M Number of instances on file.
/// @param N Number of attributes on file.

void DataSet::extractY(const char* const fileName, const unsigned int M, const unsigned int N)
{
    fstream inputFile;
    inputFile.open(fileName, ios::in);

    if(!inputFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "void extractY(const char* const, const unsigned int, const unsigned int) method" << endl
             << "Cannot open Parameter file: "<< fileName  << endl;

        exit(1);
    }

    string line;
    double X_i;
    double y;
    size_t found;

    //--Omitting lines containing '#'--//
    do
    {
        getline(inputFile, line);
        found = line.find("#");
    }while(found != string::npos);

    //--Extracting target y from file--//
    for(unsigned int m=0; m<M; m++)
    {
        stringstream ssLine(line);
        for(unsigned int n=0; n<N; n++)
        {
            ssLine >> X_i;
        }
        ssLine >> y;
        d_y(m) = y;

        getline(inputFile, line);
    }
}


// mat X(void) const method

/// Returns a matrix containing the attributes of the data set.

mat DataSet::X(void) const
{
    return d_X;
}


// mat y(void) const method

/// Returns a vector containing the targets of the data set.

vec DataSet::y(void) const
{
    return d_y;
}


// unsigned int M(void) const method

/// Returns the attribute size of the data set.

unsigned int DataSet::M(void) const
{
    return d_X.n_rows;
}


// unsigned int M(void) const method

/// Returns the target size of the data set.

unsigned int DataSet::N(void) const
{
    return d_X.n_cols;
}


// mat XTrain(void) const method

/// Returns a matrix containing the attributes of the training set.

mat DataSet::XTrain(void) const
{
    return d_X_train;
}


// mat YTrain(void) const method

/// Returns a vector containing the targets of the training set.

vec DataSet::yTrain(void) const
{
    return d_y_train;
}


// unsigned int trainingSize(void) const method

/// Returns the training data size.

unsigned int DataSet::trainingSize(void) const
{
    return d_X_train.n_rows;
}


// mat XTest(void) const method

/// Returns a matrix containing the attributes of the test set.

mat DataSet::XTest(void) const
{
    return d_X_test;
}


// mat YTest(void) const method

/// Returns a vector containing the targest of the test set.

vec DataSet::yTest(void) const
{
    return d_y_test;
}


// unsigned int testSize(void) const method

/// Returns the test data size.

unsigned int DataSet::testSize(void) const
{
    return d_X_test.n_rows;
}


// vec Mean(void) const method

/// Returns a vector containing the mean of the attributes.

vec DataSet::Mean(void) const
{
    return d_mu;
}


// vec STDEV(void) const method

/// Returns a vector containing the standard deviation of the attributes.

vec DataSet::STDEV(void) const
{
    return d_sigma;
}


// vec Min(void) const method

/// Returns a vector containing the minimum of the attributes.

vec DataSet::Min(void) const
{
    return d_min;
}


// vec Max(void) const method

/// Returns a vector containing the maximum of the attributes.

vec DataSet::Max(void) const
{
    return d_max;
}


// vec normalizeFeatures(const vec) method

/// Normalizes features of a single data instance and returns it as a vector.
/// @param x Feature vector of a single data instance.

vec DataSet::normalizeFeatures(const vec x)
{
    if(!x.n_rows)
    {
        cerr << "Regression: DataSet class." << endl
             << "vec normalizeFeatures(const vec) method" << endl
             << "Feature vector x: "<< x.n_rows  << " cannot be empty." << endl;

        exit(1);
    }

    if(d_mu.n_rows != x.n_rows)
    {
        cerr << "Regression: DataSet class." << endl
             << "vec normalizeFeatures(const vec) method" << endl
             << "Rows of feature vector x: "<< x.n_rows  << " must be equal to rows of mean vector Mu: " << d_mu.n_rows << endl;

        exit(1);
    }

    vec norm_x = x;

    //--        x_i - μ_i --//
    //--x_i <-- --------- --//
    //--           σ_i    --//
    norm_x = (norm_x - d_mu) / d_sigma;

    return norm_x;
}


// mat normalizeFeatures(const mat) method

/// Normalizes features of a data set and returns it as a matrix.
/// @param X Feature matrix were each row is an instance and each column is an attribute.

mat DataSet::normalizeFeatures(const mat X)
{
    if(!X.n_elem)
    {
        cerr << "Regression: DataSet class." << endl
             << "mat normalizeFeatures(const mat) method" << endl
             << "Matrix X: "<< X.n_elem  << " cannot be empty." << endl;

        exit(1);
    }

    if(X.n_cols != d_mu.n_rows)
    {
        cerr << "Regression: DataSet class." << endl
             << "mat normalizeFeatures(const mat) method" << endl
             << "Colums of matrix X: "<< X.n_cols  << " must be equal to rows of vector Mu: " << d_mu.n_rows << endl;

        exit(1);
    }

    mat norm_X = X;
    unsigned int n = X.n_cols;

    for(unsigned int c=0; c<n; c++)
    {
        //--        X_i - μ_i --//
        //--X_i <-- --------- --//
        //--           σ_i    --//
        norm_X.col(c) = (norm_X.col(c) - d_mu(c)) / d_sigma(c);
    }

    return norm_X;
}


// mat exponents(const mat, const unsigned int) const method

/// Calculates and returns a matrix of exponents for a given data set and the degree of polynomial for feature mapping.
/// @param X Feature matrix were each row is an instance and each column is an attribute.
/// @param degree Specifies the degree of polynomial for feature mapping.

mat DataSet::exponents(const mat X, const unsigned int degree) const
{
    unsigned int n = X.n_cols;
    double exp_sum;

    mat exp;
    vec v = linspace<vec>(0, degree, degree + 1);

    exp.insert_cols(0, v);

    for(unsigned int c=1; c<n; c++)
    {
        mat subspace_exp = exp;

        vec v = zeros<vec>(exp.n_rows);
        exp.insert_cols(0, v);

        for(unsigned int d=1; d<=degree; d++)
        {
            vec v = ones<vec>(subspace_exp.n_rows) * d;
            subspace_exp.insert_cols(0, v);

            exp.insert_rows(exp.n_rows, subspace_exp);

            subspace_exp.shed_col(0);
        }
    }

    for(unsigned int r=0; r<exp.n_rows;)
    {
        exp_sum = sum(exp.row(r));
        if(exp_sum > degree || exp_sum == 0)
        {
            exp.shed_row(r);
        }
        else
        {
            r++;
        }
    }

    return exp;
}


// mat mapFeatures(const mat, const unsigned int) const method

/// Performs feature mapping and returns a matrix containing the original features plus new features.
/// @param X Feature matrix were each row is an instance and each column is an attribute.
/// @param degree Specifies the degree of polynomial for feature mapping. degree ≥ 1. degree = 1 ensures data set remains unchanged.

mat DataSet::mapFeatures(const mat X, const unsigned int degree) const
{

    if(degree == 0)
    {
        cerr << "Regression: DataSet class." << endl
             << "mat mapFeatures(const mat, const unsigned int) const method." << endl
             << "Parameter degree: " << degree << " for polynomial feature mapping has to be >= 1 "
             << endl;

        exit(1);
    }

    mat out;

    unsigned int m = X.n_rows;
    unsigned int n = X.n_cols;

    mat exp = exponents(X, degree);

    for(unsigned int r=0; r<exp.n_rows; r++)
    {
        vec  v = ones<vec>(m);

        for(unsigned int c=0; c<n; c++)
        {
            v = v % pow(X.col(c), exp(r,c));
        }
        out.insert_cols(out.n_cols, v);
    }

    return out;
}


void DataSet::segmentDataSet(const double trainPercent, const double testPercent)
{
    if(trainPercent + testPercent != 100.0)
    {
        cerr << "Regression: DataSet class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training set: " << trainPercent << " + Test set: " << testPercent << " != 100%"
             << endl;

        exit(1);
    }

    unsigned int m = d_X.n_rows;
    unsigned int n = d_X.n_cols;

    //--Combine matrix X and vector y by inserting vector y as the last column of matrix X--//
    mat Xy = d_X;
    Xy.insert_cols(n, d_y);

    //--Shuffle the whole data set--//
    Xy = shuffle(Xy);

    //--Calculate training set and test set size--//
    unsigned int trainSize = m * (trainPercent/100.0);
    unsigned int testSize = m - trainSize;

    //--Segregate data into training and test sets--//
    d_X_train = Xy;
    if(testSize)
    {
        d_X_train.shed_rows(trainSize, m-1);
    }
    d_y_train = d_X_train.col(n);
    d_X_train.shed_col(n);


    if(testSize)
    {
        d_X_test = Xy;
        d_X_test.shed_rows(0, trainSize-1);

        d_y_test = d_X_test.col(n);
        d_X_test.shed_col(n);
    }

    if(d_X_train.n_rows + d_X_test.n_rows != m)
    {
        cerr << "Regression: DataSet class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training set: " << d_X_train.n_rows << " + Test set: " << d_X_train.n_rows << " != Data size: " << m
             << endl;

        exit(1);
    }

    cout << endl << "Training set size: " << d_X_train.n_rows
         << endl << "Test set size: " << d_X_test.n_rows << endl;
}


void DataSet::printDataSet(void) const
{
    //--Combine matrix X and vector y by inserting vector y as the last column of X--//
    mat Xy = d_X;
    Xy.insert_cols(Xy.n_cols, d_y);

    cout << endl << "Data set:" << endl;
    Xy.print();
}


void DataSet::printTrainingSet(void) const
{
    //--Combine matrix XTrain and vector yTrain by inserting vector yTrain as the last column of XTrain--//
    mat Xy = d_X_train;
    Xy.insert_cols(Xy.n_cols, d_y_train);

    cout << endl << "Training set:" << endl;
    Xy.print();
}


void DataSet::printTestSet(void) const
{
    //--Combine matrix XTest and vector yTest by inserting vector yTest as the last column of XTest--//
    mat Xy = d_X_test;
    Xy.insert_cols(Xy.n_cols, d_y_test);

    cout << endl << "Test set:" << endl;
    Xy.print();
}

void DataSet::saveToFile(const mat A) const
{
    fstream outputFile;
    remove("../Output/datafile.dat");
    outputFile.open("../Output/datafile.dat", ios_base::out);

    for(unsigned int row=0; row<A.n_rows; row++)
    {
        for(unsigned int col=0; col<A.n_cols; col++)
        {
            outputFile << A(row,col) << " ";
        }
        outputFile << endl;
    }

    outputFile.close();
}
