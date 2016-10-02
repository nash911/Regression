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
/// @param fileName Path and name of the file containing the training data.
/// @param degree Specifies the degree of polynomial for feature mapping. degree ≥ 1. degree = 1 ensures data set remains unchanged.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.
/// @param MNIST Indicates if dataset is MNIST or not.

DataSet::DataSet(const char* fileName, const unsigned int degree=1, const double trainPercent=70, const double testPercent=30, const bool MNIST=false)
{
    if(!fileName)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double, const bool) constructor." << endl
             << "Cannot open data file: " << fileName
             << endl;

        exit(1);
    }

    if(degree == 0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double, const bool) constructor." << endl
             << "Parameter degree: " << degree << " for polynomial feature mapping has to be >= 1 "
             << endl;

        exit(1);
    }

    if(trainPercent <= 0.0 || testPercent < 0.0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double, const bool) constructor." << endl
             << "Training set = " << trainPercent << "% has to be > 0% and Test set = "<< testPercent << "% has to be >= 0%."
             << endl;

        exit(1);
    }

    if(trainPercent + testPercent != 100.0)
    {
        cerr << "Regression: DataSet class." << endl
             << "DataSet(const char*, const unsigned int, const double, const double, const bool) constructor." << endl
             << "Training set = " << trainPercent << " + Test set = "<< testPercent << " has to be equal to 100%."
             << endl;

        exit(1);
    }

    if(MNIST)
    {
        string filePath(fileName);
        size_t found = filePath.find_last_of("/");
        filePath = filePath.substr(0,found+1);

        extractMNISTData(filePath);
    }
    else
    {
        extractDataFromFile(fileName, degree, trainPercent, testPercent);
    }
}


// void extractMNISTData(const string) method.

/// Extracts training and test data and the respective labels from MNIST dataset, the path of which is passed as a parameter.
/// Encodes labels as one-hot vector format.
/// Normalizes the training and test dat sets.
/// @param filePath Path of the file containing MNIST dataset.

void DataSet::extractMNISTData(const string filePath)
{
    string train_img = filePath + "train-images.idx3-ubyte";
    string train_label = filePath + "train-labels.idx1-ubyte";

    string test_img = filePath + "t10k-images.idx3-ubyte";
    string test_label = filePath + "t10k-labels.idx1-ubyte";

    fstream dataFile;

    cout << endl << "   MNIST data set" << endl << "Training image file: " << train_img << endl;
    cout << "Training labels file: " << train_label << endl;
    cout << "Test image file: " << test_img << endl;
    cout << "Test labels file: " << test_label << endl;

    //--Extract training data and labels--//
    extractMNISTimg(train_img, d_train_img_cube);
    extractMNISTlabel(train_label, d_train_label_vec);

    //--Extract training data and labels--//
    extractMNISTimg(test_img, d_test_img_cube);
    extractMNISTlabel(test_label, d_test_label_vec);

    //--Extract unique labels and sort them--//
    d_class = sort(unique(d_train_label_vec));

    //--Encode training and test labels into one-hot format--//
    oneHotEncode(d_train_label_vec, d_train_1hot_mat);
    oneHotEncode(d_test_label_vec, d_test_1hot_mat);

    //--Normalize traininga and test data--//
    normalizeFeatures(d_train_img_cube);
    normalizeFeatures(d_test_img_cube);

    unrollCubetoMatrix(d_train_img_cube, d_X_train);
    unrollCubetoMatrix(d_test_img_cube, d_X_test);

    cout << endl << "Number of training instances: " << d_X_train.n_rows << endl;
    cout << endl << "Number of test instances: " << d_X_test.n_rows << endl;
    cout << endl << "Number of attributes per instance: " << d_X_train.n_cols << endl;
    cout << endl << "Label class vector:" << d_class.t();
}


// int ReverseInt(int) method.

/// Source: http://eric-yuan.me/cpp-read-mnist/

int DataSet::ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


// void extractMNISTimg(const string, cube&) method.

/// Extracts MNIST image data from the file whose path and name is passed as a parameter.
/// Source: http://eric-yuan.me/cpp-read-mnist/
/// @param fileName Path and name of the file containing the image data.
/// @param tensor Reference of Armadillo::cube object to extract image data into.

void DataSet::extractMNISTimg(const string fileName, cube &tensor)
{
    ifstream dataFile (fileName.c_str(), ios::binary);
    if (!dataFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "void extractMNISTimg(const string, cube&) method." << endl
             << "Unable to open image data file: "<< fileName
             << endl;

        exit(1);
    }
    else
    {
        int magic_number = 0;
        unsigned int number_of_images = 0;
        unsigned int n_rows = 0;
        unsigned int n_cols = 0;

        dataFile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        dataFile.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        dataFile.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        dataFile.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        tensor.set_size(n_rows, n_cols, number_of_images);

        for(unsigned int i = 0; i < number_of_images; i++)
        {
            mat tp(n_rows, n_cols);
            for(unsigned int r = 0; r < n_rows; r++)
            {
                for(unsigned int c = 0; c < n_cols; c++)
                {
                    unsigned char temp = 0;
                    dataFile.read((char*) &temp, sizeof(temp));
                    tensor(r,c,i) = (double) temp;
                }
            }
        }
    }
}


// void extractMNISTlabel(const string, vec&) method.

/// Extracts MNIST label data from the file whose path and name is passed as a parameter.
/// Source: http://eric-yuan.me/cpp-read-mnist/
/// @param fileName Path and name of the file containing the label data.
/// @param label Reference of Armadillo::vec object to extract image data into.

void DataSet::extractMNISTlabel(const string fileName, vec &label)
{
    ifstream dataFile (fileName.c_str(), ios::binary);
    if (!dataFile.is_open())
    {
        cerr << "Regression: DataSet class." << endl
             << "void extractMNISTlabel(const string, ved&) method." << endl
             << "Unable to open image data file: "<< fileName
             << endl;

        exit(1);
    }
    else
    {
        int magic_number = 0;
        unsigned int number_of_images = 0;

        dataFile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        dataFile.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        label.set_size(number_of_images);

        for(unsigned int i = 0; i < number_of_images; i++)
        {
            unsigned char temp = 0;
            dataFile.read((char*) &temp, sizeof(temp));
            label(i)= (double)temp;
        }
    }
}


// void oneHotEncode(const vec, mat&) method

/// Based on the labels vector y ∈ R^m, extracts k unique target values.
/// Creates matrix Y ∈ R^(kxm), where y⁽i⁾ ∈ R^k, and of type [... 0 1 0 ...]'
/// @param labels Vector containing instance labels.
/// @param oneHotMat Reference of Armadillo::mat object to hold encoded one-hot vectors.

void DataSet::oneHotEncode(const vec labels, mat &oneHotMat)
{
    //vec uniqueLabels = unique(labels);
    if(d_class.is_empty())
    {
        cerr << "Regression: DataSet class." << endl
             << "void oneHotEncode(const vec, mat&) method." << endl
             << "Labels class vector cannot be empty."
             << endl;

        exit(0);
    }

    unsigned int instSize = labels.n_rows;
    oneHotMat.set_size(d_class.n_rows, instSize);
    oneHotMat.zeros();

    for(unsigned int i=0; i<instSize; i++)
    {
        uvec indx = find(d_class == labels[i], 1);
        oneHotMat(indx[0], i) = 1.0;
    }
}


// void unrollCubetoMatrix(const cube&, mat&) method

/// Unroll a cube into a matrix.
/// Each slice of the cube is unroller into a row vector of the resulting matrix.
/// @param tensor Reference to 3D Cube containing instances in the form of 2D matrices.
/// @param dataset Reference of Armadillo::mat object to hold unrolled data in the form of row vectors.

void DataSet::unrollCubetoMatrix(const cube &tensor, mat &dataset)
{
    unsigned int instSize = tensor.n_slices;
    unsigned int rows = tensor.n_rows;
    unsigned int cols = tensor.n_cols;

    dataset.set_size(instSize, (rows*cols));

    for(unsigned int i=0; i<instSize; i++)
    {
        dataset.row(i) = vectorise(tensor.slice(i), 1);
    }
}


// void extractDataFromFile(const char*, const unsigned int, const double, const double)

/// Extracts training data containing features and target from the file whose path and name is passed as a parameter.
/// Creates new polynomial features through feature mapping.
/// Calculates the mean and standard deviation of the data features and then the features are Mean Normalized.
/// Shuffels the data set and divides it into training and test sets.
/// @param fileName Path and name of the file containing the training data.
/// @param degree Specifies the degree of polynomial for feature mapping. degree ≥ 1. degree = 1 ensures data set remains unchanged.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.

void DataSet::extractDataFromFile(const char* fileName, const unsigned int degree, const double trainPercent, const double testPercent)
{
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

    //--Extract unique labels and sort them--//
    d_class = sort(unique(d_y));

    if(d_X.n_rows != d_y.n_rows)
    {
        cerr << "Regression: DataSet class." << endl
             << "void extractDataFromFile(const char*, const unsigned int, const double, const double) method." << endl
             << "No. of instances in matrix X: " << d_X.n_rows << "  and vector y: " << d_y.n_rows << " do not match."
             << endl;

        exit(0);
    }

    //--Create new features through Feature Mapping--//
    d_X = mapFeatures(d_X, degree);

    //--Calculate and store the μ and σ of the features--//
    d_mu = mean(d_X).t();
    d_sigma = stddev(d_X).t();

    //--Calculate and store the min and max of the features--//
    d_min = min(d_X).t();
    d_max = max(d_X).t();

    //--Normalize features--//
    d_X = normalizeFeatures(d_X);

    //--Shuffle the data and segment into training and test sets--//
    segmentDataSet(trainPercent, testPercent);

    //--Encode train and test lables into one-hot format--//
    oneHotEncode(d_y_train, d_train_1hot_mat);
    oneHotEncode(d_y_test, d_test_1hot_mat);
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


// vec y(void) const method

/// Returns a vector containing the targets of the data set.

vec DataSet::y(void) const
{
    return d_y;
}


// vec labels(void) const method

/// Returns a vector containing the k distinct labels of the data set.

vec DataSet::labels(void) const
{
    return d_class;
}


// unsigned int M(void) const method

/// Returns the instance size of the data set.

unsigned int DataSet::M(void) const
{
    return d_X.n_rows;
}


// unsigned int N(void) const method

/// Returns the attribute size of the training set.

unsigned int DataSet::N(void) const
{
    return d_X_train.n_cols;
}


// unsigned int K(void) const method

/// Returns the class size of the data set.

unsigned int DataSet::K(void) const
{
    return d_class.n_rows;
}


// mat& XTrain(void) method

/// Returns reference to a matrix containing the instances of the training set.

mat& DataSet::XTrain(void)
{
    return d_X_train;
}


// mat& YTrain(void) method

/// Returns reference to a matrix of size Mx1, containing targets of the training set.

mat& DataSet::yTrain(void)
{
    return d_y_train;
}


// mat& Train_oneHotMatrix(void) method

/// Returns reference to a matrix containing targets of the training set, in the form of one-hot vector format.

mat& DataSet::Train_oneHotMatrix(void)
{
    return d_train_1hot_mat;
}


// unsigned int trainingSize(void) const method

/// Returns the training data size.

unsigned int DataSet::trainingSize(void) const
{
    return d_X_train.n_rows;
}


// mat& XTest(void) method

/// Returns reference to a matrix containing instances of the test set.

mat& DataSet::XTest(void)
{
    return d_X_test;
}


// mat& YTest(void) method

/// Returns reference to a matrix of size Mx1, containing targest of the test set.

mat& DataSet::yTest(void)
{
    return d_y_test;
}


// mat& Test_oneHotMatrix(void) method

/// Returns reference to a matrix containing targets of the test set, in the form of one-hot vector format.

mat& DataSet::Test_oneHotMatrix(void)
{
    return d_test_1hot_mat;
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


// void normalizeFeatures(cube&) method

/// Normalizes features of a data set in the reference Armadillo::cube object.
/// @param X Reference of Armadillo::cube object, were each slice is an instance.

void DataSet::normalizeFeatures(cube &X)
{
    if(!X.n_elem)
    {
        cerr << "Regression: DataSet class." << endl
             << "void normalizeFeatures(cube&) method" << endl
             << "Cube X: "<< X.n_elem  << " cannot be empty." << endl;

        exit(1);
    }

    double min = X.min();
    double max = X.max();
    double mid = (max - min) / 2.0;

    cout << endl << "Min: " << min << "  Max: " << max << "  Mid: " << mid << endl;

    unsigned int rows = X.n_rows;
    unsigned int cols = X.n_cols;
    unsigned int slice = X.n_slices;

    for(unsigned int s=0; s<slice; s++)
    {
        for(unsigned int r=0; r<rows; r++)
        {
            for(unsigned int c=0; c<cols; c++)
            {
                X(r,c,s) = (X(r,c,s) - mid) / max;
            }
        }
    }
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


// void segmentDataSet(const double, const double) const method

/// Shuffels the data set and divides it into training and test sets.
/// @param trainPercent Training split of the data set > 0%.
/// @param testPercent Test split of the data set ≥ 0%.

void DataSet::segmentDataSet(const double trainPercent, const double testPercent)
{
    if(!(trainPercent >= 0.0 && trainPercent <= 100.0) || !(testPercent >= 0.0 && testPercent <= 100.0))
    {
        cerr << "Regression: DataSet class." << endl
             << "void segmentDataSet(const double, const double) method" << endl
             << "Training size(%): " << trainPercent << " and Test size(%): " << testPercent << " should both be in the range [0,100]."
             << endl;

        exit(1);
    }

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
    if(trainSize)
    {
        d_X_train = Xy;
        d_X_train.shed_rows(trainSize, m-1);

        d_y_train = d_X_train.col(n);
        d_X_train.shed_col(n);
    }

    if(testSize)
    {
        d_X_test = Xy;
        if(trainSize)
        {
            d_X_test.shed_rows(0, trainSize-1);
        }

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


// void printDataSet(void) const method

/// Print the combined training and test data set.

void DataSet::printDataSet(void) const
{
    //--Combine matrix X and vector y by inserting vector y as the last column of X--//
    mat Xy = d_X;
    Xy.insert_cols(Xy.n_cols, d_y);

    cout << endl << "Data set:" << endl;
    Xy.print();
}


// void printTrainingSet(void) const method

/// Print the training data set.

void DataSet::printTrainingSet(void) const
{
    //--Combine matrix XTrain and vector yTrain by inserting vector yTrain as the last column of XTrain--//
    mat Xy = d_X_train;
    Xy.insert_cols(Xy.n_cols, d_y_train);

    cout << endl << "Training set:" << endl;
    Xy.print();
}


// void printTestSet(void) const method

/// Print the test data set.

void DataSet::printTestSet(void) const
{
    //--Combine matrix XTest and vector yTest by inserting vector yTest as the last column of XTest--//
    mat Xy = d_X_test;
    Xy.insert_cols(Xy.n_cols, d_y_test);

    cout << endl << "Test set:" << endl;
    Xy.print();
}


// void saveToFile(const mat) const method

/// Save a matrix to a file in matrix format.
/// @param A Matrix to be saved.

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
