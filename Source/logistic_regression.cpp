/********************************************************************************************/
/*                                                                                          */
/*   Regression: A C++ library for Linear and Logistic Regression.                          */
/*                                                                                          */
/*   L O G I S T I C   R E G R E S S I O N   C L A S S                                      */
/*                                                                                          */
/*   Avinash Ranganath                                                                      */
/*   Robotics Lab, Department of Systems Engineering and Automation                         */
/*   University Carlos III of Mardid(UC3M)                                                  */
/*   Madrid, Spain                                                                          */
/*   E-mail: nash911@gmail.com                                                              */
/*   https://sites.google.com/site/anashranga/                                              */
/*                                                                                          */
/********************************************************************************************/

#include "logistic_regression.h"


LogisticRegression::LogisticRegression(const DataSet& ds):Regression(ds)
{
    d_classification_threshold = 0.5;
}


double LogisticRegression::h_Theta(vec x) const
{
    if(x.n_rows != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double h_Theta(vec) const method" << endl
             << "Size of vectors x: "<< x.n_rows  << " and Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    //--Insert 1.0 to the first row of the data vector--//
    x.insert_rows(0,1);
    x(0) = 1.0;

    //--h_Ө(x) = sigmoid(Ө'x)--//
    return sigmoid(d_Theta.t() * x)(0);
}


double LogisticRegression::cost(mat X, const vec y) const
{
    if(X.n_rows != y.n_rows)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double cost(mat, const vec) const method" << endl
             << "Rows of matrix X: "<< X.n_rows  << " must be equal to rows of vector y: " << y.n_rows << endl;

        exit(1);
    }

    if(X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double cost(mat, const vec) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    double m = X.n_rows;

    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    vec cost;
    vec sig = sigmoid(X * d_Theta);

    vec theta = d_Theta;
    theta(0) = 0;

    //--        1  m                                                      λ   n       --//
    //--J(Ө) = --- ∑ [-y⁽i⁾ log(h_Ө(x⁽i⁾)) - (1-y⁽i⁾)log(1-h_Ө(x⁽i⁾))] + ---- ∑(Ө_j)^2--//
    //--        m  i                                                      2m  j       --//
    cost = ((1.0/m) * sum((((-1.0) * y) % log(sig)) - ((1.0 - y) % log(1.0 - sig)))) + ((d_lamda / (2.0*m)) * (sum(theta % theta)));

    return(cost(0));
}


double LogisticRegression::gradientdescent(const double delta)
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
        vec theta = d_Theta;
        theta(0) = 0;

        //--                  _                              _                --//
        //--               1 |  m                             |               --//
        //--Ө_j := Θ_j - 𝛼---|  ∑ [h_Ө(x⁽i⁾) - y⁽i⁾] (x_j)⁽i⁾ |        ∀ j = 0--//
        //--               m |_ i                            _|               --//

        //--                  _                                     _          --//
        //--               1 |  m                                    |         --//
        //--Ө_j := Θ_j - 𝛼---|  ∑ [h_Ө(x⁽i⁾) - y⁽i⁾] (x_j)⁽i⁾ + λӨ_j | ∀ j >= 1--//
        //--               m |_ i                                   _|         --//

        d_Theta = d_Theta - ((d_alpha/(double)m) * ((X.t() * (sigmoid(X * d_Theta) - y)) + (d_lamda * theta)));

        c_prev = c;
        c = cost(d_dset.XTrain(), d_dset.yTrain());

        costGraph << it++ << " " << c << endl;

    }while(fabs(c_prev - c) > delta);

    cout << endl << "Finished training. Training details:"
         << endl << "Iterations: " << it
         << endl << "Delta_J(Theta): " << fabs(c_prev - c)
         << endl << "J(Theta): " << c << endl;

    costGraph.close();

    return c;
}


double LogisticRegression::classificationThreshold(void) const
{
    return d_classification_threshold;
}


void LogisticRegression::set_classificationThreshold(const double classification_threshold)
{
    if(classification_threshold < 0.0 || classification_threshold > 1.0)
    {
        cerr << "Regression: Logistic Regression class." << endl
             << "void set_classificationThreshold(const double) method" << endl
             << "classification_threshold: "<< classification_threshold  << " must be >= 0 and <= 1.0" << endl;

        exit(1);
    }
    else
    {
        d_classification_threshold = classification_threshold;
    }
}


vec LogisticRegression::sigmoid(const vec z) const
{
    //--     1      --//
    //-- ---------- --//
    //-- 1 + e^(-z) --//
    return (1.0 / (1.0 + exp((-1.0) * z)));
}


uvec LogisticRegression::predict(mat X) const
{
    if(X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "uvec predict(mat) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    unsigned int m = X.n_rows;

    vec X_0 = ones<vec>(m);
    X.insert_cols(0, X_0);

    vec y;
    y = sigmoid(X * d_Theta);

    uvec bool_y = (y >= d_classification_threshold);

    return bool_y;
}


mat LogisticRegression::confusionMatrix(const mat X, const vec y) const
{
    mat confMat = zeros<mat>(2,2);
    uvec predicted_y = predict(X);

    unsigned int TP = sum((predicted_y + y) == 2);
    unsigned int TN = sum((predicted_y + y) == 0);
    unsigned int FP = sum((predicted_y - y) == 1);
    unsigned int FN = sum((predicted_y - y) == -1);

    confMat(0,0) = TP;
    confMat(0,1) = FP;
    confMat(1,0) = FN;
    confMat(1,1) = TN;

    return confMat;
}


void LogisticRegression::print_confusionMatrix(const mat confMat) const
{
    double TP = confMat(0,0);
    double TN = confMat(1,1);
    double FP = confMat(0,1);
    double FN = confMat(1,0);

    cout << endl << "       Confusion Matrix        "
         << endl << "     --------------------      "
         << endl << "True Positive    False Positive"
         << endl << "-------------    --------------"
         << endl << "    " << TP << "                " << FP << endl
         << endl << "    " << FN << "                " << TN
         << endl << "--------------    -------------"
         << endl << "False Negative    True Negative" << endl;

    cout << endl << "No. of actual positives: " << TP + FN
         << endl << "No. of actual negatives: " << TN + FP
         << endl << "Total Misclassifications: " << FP + FN << endl;
}


double LogisticRegression::f1Score(const bool show_stats=false) const
{
    mat confMat = confusionMatrix(d_dset.XTest(), d_dset.yTest());

    if(show_stats)
    {
        print_confusionMatrix(confMat);
    }

    double TP = confMat(0,0);
    double TN = confMat(1,1);
    double FP = confMat(0,1);
    double FN = confMat(1,0);

    double precision;
    double recall;
    double specificity; //--True negative rate--//
    double accuracy;

    double f1_score;

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);

    specificity = TN / (TN + FP);
    accuracy = (TP + TN) / (TP + TN + FP + FN);

    if(show_stats)
    {
        cout << endl << "Precision:   " << precision
             << endl << "Recall:      " << recall
             << endl << "Specificity: " << specificity
             << endl << "Accuracy:    " << accuracy << endl;
    }

    f1_score = 2.0 * ((precision * recall) / (precision + recall));

    return f1_score;
}


double LogisticRegression::f1Score(const mat X, const vec y, const bool show_stats=false) const
{
    mat confMat = confusionMatrix(X, y);

    if(show_stats)
    {
        print_confusionMatrix(confMat);
    }

    double TP = confMat(0,0);
    double TN = confMat(1,1);
    double FP = confMat(0,1);
    double FN = confMat(1,0);

    double precision;
    double recall;
    double specificity; //--True negative rate--//
    double accuracy;

    double f1_score;

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);

    specificity = TN / (TN + FP);
    accuracy = (TP + TN) / (TP + TN + FP + FN);

    if(show_stats)
    {
        cout << endl << "Precision:   " << precision
             << endl << "Recall:      " << recall
             << endl << "Specificity: " << specificity
             << endl << "Accuracy:    " << accuracy << endl;
    }

    f1_score = 2.0 * ((precision * recall) / (precision + recall));

    return f1_score;
}
