#include "dataset.h"
//#include "regression.h"
#include "linear_regression.h"
#include "logistic_regression.h"

#define ALPHA 0.01
#define LAMDA 1.0

#define DEGREE 5

#define TRAIN_PERCENT 70
#define TEST_PERCENT 30

#define DELTA 0.0000001
#define MAX_ITERATIONS 1000

void linear_regression(char* fileName=NULL)
{
    char* dataFileName;

    if(fileName != NULL)
    {
        dataFileName = fileName;
    }
    else
    {
        dataFileName = "../Data/servo.dat";
    }

    DataSet d(dataFileName, DEGREE, TRAIN_PERCENT, TEST_PERCENT, false);

    LinearRegression linR(d);

    linR.set_lamda(LAMDA);
    linR.set_alpha(ALPHA);

    linR.gradientdescent(d.XTest(), d.yTest(), DELTA, MAX_ITERATIONS);

    cout << "Cost on test set: " << linR.cost(d.XTest(), d.yTest()) << endl << endl;

    //linR.create_model(DEGREE);
}


void logistic_regression(char* fileName=NULL, const bool MNIST=false)
{
    char* dataFileName;

    if(fileName != NULL)
    {
        dataFileName = fileName;
    }
    else
    {
        dataFileName = "../Data/chip.dat";
    }

    DataSet d(dataFileName, DEGREE, TRAIN_PERCENT, TEST_PERCENT, MNIST);

    cout << endl << "Data set size: " << d.X().n_rows << "x" << d.X().n_cols << endl;

    LogisticRegression logR(d);

    logR.set_lamda(LAMDA);
    logR.set_alpha(ALPHA);

    logR.gradientdescent(d.XTrain(), d.Train_oneHotMatrix(), DELTA, MAX_ITERATIONS);

    cout << endl << "F1_Score: " << logR.f1Score(d.XTest(), d.Test_oneHotMatrix(), true) << endl;
}

int main(int argc, char* argv[])
{
    //--Initializing random seed--//
    srand (time(NULL));

    char* dataFileName;
    fstream dataFile;
    bool MNIST = false;

    if(argc >= 2)
    {
        dataFileName = argv[1];

        dataFile.open(dataFileName, ios_base::in);
        if(!dataFile.is_open())
        {
            cerr << "Regression: Main." << endl
                 << "int main(int, char*) method" << endl
                 << "Cannot open data file: "<< dataFileName
                 << endl;

            exit(1);
        }
        dataFile.close();

        if(!strcmp(argv[2], "-MNIST"))
        {
            MNIST = true;
        }
    }
    else
    {
        dataFileName = NULL;
    }

    //linear_regression(dataFileName);
    logistic_regression(dataFileName, MNIST);

    return 0;
}
