#include "dataset.h"
//#include "regression.h"
#include "linear_regression.h"
#include "logistic_regression.h"

#define ALPHA 0.01
#define LAMDA 1.0

#define DEGREE 4

#define TRAIN_PERCENT 70
#define TEST_PERCENT 30

#define DELTA 0.000001

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

    linR.gradientdescent(DELTA);

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

    LogisticRegression logR(d);

    /*logR.set_lamda(LAMDA);
    logR.set_alpha(ALPHA);

    logR.gradientdescent(DELTA);

    vec lamda(10);
    lamda(0) = 0.0;
    lamda(1) = 0.1;
    lamda(2) = 0.3;
    lamda(3) = 0.6;
    lamda(4) = 1.0;
    lamda(5) = 3.0;
    lamda(6) = 6.0;
    lamda(7) = 10.0;
    lamda(8) = 30.0;
    lamda(9) = 60.0;


    for(unsigned int i=0; i<lamda.n_rows; i++)
    {
        logR.init_theta();
        logR.set_lamda(lamda(i));
        logR.gradientdescent(DELTA);
        cout << endl << "Test - " << i+1 << ": F1_Score: " << logR.f1Score(true) << endl;
    }*/
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
