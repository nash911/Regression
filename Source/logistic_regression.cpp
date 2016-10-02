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


LogisticRegression::LogisticRegression(const DataSet& ds):Regression(ds, "Classification")
{
    d_class_func = Softmax;
    d_classification_threshold = 0.5;
}


vec LogisticRegression::h_Theta(vec x) const
{
    if(x.n_rows != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "vec h_Theta(vec) const method" << endl
             << "Size of vectors x: "<< x.n_rows  << " and Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    //--Insert 1.0 to the first row of the data vector--//
    x.insert_rows(0,1);
    x(0) = 1.0;

    if(d_class_func == Sigmoid)
    {
        //--h_Ө(x) = sigmoid(Ө'x)--//
        return sigmoid(d_Theta.t() * x);
    }
    else if(d_class_func == Softmax)
    {
        //--h_Ө(x) = softmax(Ө'x)--//
        return softmax(d_Theta.t() * x);
    }
    else
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "vec h_Theta(vec) const method" << endl
             << "Invalid classification function type: "<< d_class_func  << endl;

        exit(1);
    }
}


double LogisticRegression::cost(mat& X, const mat& Y) const
{
    if(X.n_rows != Y.n_cols)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double cost(mat&, const vec&) const method" << endl
             << "Rows of matrix X: "<< X.n_rows  << " must be equal to cols of KxM matrix Y: " << Y.n_cols << endl;

        exit(1);
    }


    if(X.n_cols != d_Theta.n_rows && X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double cost(mat&, const mat&) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    double m = X.n_rows;
    mat cost;
    bool bias_term_added = false;

    if(X.n_cols == d_Theta.n_rows-1)
    {
        vec X_0 = ones<vec>(m);
        X.insert_cols(0, X_0);

        bias_term_added = true;
    }

    mat h_theta;
    if(d_class_func == Sigmoid)
    {
        h_theta = sigmoid(X * d_Theta);
    }
    else if(d_class_func == Softmax)
    {
        h_theta = softmax(X * d_Theta);
    }
    else
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "double cost(mat, const mat) const method" << endl
             << "Invalid classification function type: "<< d_class_func  << endl;

        exit(1);
    }

    mat theta = d_Theta;
    theta.row(0).zeros();

    //--        1  m                                 --//
    //--J(Ө) = --- ∑ [-y'⁽i⁾ log(h_Ө(x⁽i⁾))], ∀ j = 0--//
    //--        m  i                                 --//

    //--        1  m                            λ   n                 --//
    //--J(Ө) = --- ∑ [-y'⁽i⁾ log(h_Ө(x⁽i⁾))] + ---- ∑(Ө_j)^2, ∀ j >= 1--//
    //--        m  i                            2m  j                 --//

    cost = ((-1.0/m) * accu(Y.t() % log(h_theta))) + ((d_lamda / (2.0*m)) * (accu(theta % theta)));

    if(bias_term_added)
    {
        X.shed_col(0);
    }

    return(cost(0,0));
}


mat LogisticRegression::derivative(const mat& X, const mat& Y) const
{
    mat DeltaTheta;
    mat theta = d_Theta;
    theta.row(0).zeros();

    mat h_theta;

    if(d_class_func == Sigmoid)
    {
        h_theta = sigmoid(X * d_Theta);
    }
    else if(d_class_func == Softmax)
    {
        h_theta = softmax(X * d_Theta);
    }
    else
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "mat derivative(const mat&, const mat&) const method" << endl
             << "Invalid classification function type: "<< d_class_func  << endl;

        exit(1);
    }

    //--            _                              _          --//
    //--  ∂J(Ө)    |  m                             |         --//
    //-- ------- = |  ∑ [h_Ө(x⁽i⁾) - y⁽i⁾] (x_j)⁽i⁾ |, ∀ j = 0--//
    //--   ∂Θ_j    |_ i                            _|         --//

    //--            _                                     _           --//
    //--  ∂J(Ө)    |  m                                    |          --//
    //-- ------- = |  ∑ [h_Ө(x⁽i⁾) - y⁽i⁾] (x_j)⁽i⁾ + λӨ_j |, ∀ j >= 1--//
    //--   ∂Θ_j    |_ i                                   _|          --//

    DeltaTheta = (X.t() * (h_theta - Y.t())) + (d_lamda * theta);

    return DeltaTheta;
}


string LogisticRegression::classificationFunction(void) const
{
    switch(d_class_func)
    {
    case Sigmoid:
    {
        return("Sigmoid");
    }
    case Softmax:
    {
        return("Softmax");
    }
    default:
    {
        cerr << "Regression: Logistic Regression class." << endl
             << "string classificationFunction(void) method" << endl
             << "Invalid classification function type: "<< d_class_func  << endl;

        exit(1);
    }
    }
}


void LogisticRegression::set_classificationFunction(const string& class_func)
{
    if(class_func == "Sigmoid")
    {
        d_class_func = Sigmoid;
    }
    else if(class_func == "Softmax")
    {
        d_class_func = Softmax;
    }
    else
    {
        cerr << "Regression: Logistic Regression class." << endl
             << "void set_classificationFunction(const string&) method" << endl
             << "Invalid classification function type: "<< class_func  << endl;

        exit(1);
    }
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


mat LogisticRegression::sigmoid(const mat z) const
{
    //--     1      --//
    //-- ---------- --//
    //-- 1 + e^(-z) --//

    return (1.0 / (1.0 + exp((-1.0) * z)));
}


mat LogisticRegression::softmax(const mat z) const
{
    //--   e^(z_i)   --//
    //-- ----------- --//
    //-- ∑_j e^(z_j) --//

    mat e_z = exp(z);
    vec sum_scores;
    mat denom;

    if(e_z.n_cols > 1)
    {
        sum_scores = sum(e_z,1);
        denom = repmat(sum_scores, 1, e_z.n_cols);
    }
    else
    {
        sum_scores = sum(e_z,0);
        denom = repmat(sum_scores, e_z.n_rows, 1);
    }

    return(e_z / denom);
}


mat LogisticRegression::predict(mat X, const mat target) const
{
    if(X.n_cols != d_Theta.n_rows-1)
    {
        cerr << "Regression: LogisticRegression class." << endl
             << "mat predict(const mat, const mat) const method" << endl
             << "Colum size of matrix X: "<< X.n_cols  << " and size of vector Theta: " << d_Theta.n_rows << " are incompatable." << endl;

        exit(1);
    }

    unsigned int instSize = X.n_rows;

    vec X_0 = ones<vec>(instSize);
    X.insert_cols(0, X_0);

    mat H;
    H = sigmoid(X * d_Theta);

    mat Y = zeros<mat>(instSize, target.n_rows);

    ucolvec max_indx = index_max(H,1);

    for(unsigned int i=0; i<instSize; i++)
    {
        Y(i, max_indx[i]) = 1.0;
    }

    return Y;
}


umat LogisticRegression::confusionMatrix(const mat X, const mat labels) const
{
    umat confMat(d_dset.K(), d_dset.K());
    confMat.zeros();

    mat predicted_y = predict(X, labels);

    unsigned int instSize = X.n_rows;
    uvec row_indx;
    uvec col_indx;

    for(unsigned int i=0; i<instSize; i++)
    {
        row_indx = find(labels.col(i) == 1, 1);
        col_indx = find(predicted_y.row(i) == 1, 1);

        confMat(row_indx, col_indx) += 1;
    }

    urowvec col_sum = sum(confMat, 0);
    ucolvec row_sum = sum(confMat, 1);

    uvec TP = confMat.diag();
    uvec TN = accu(confMat) - (col_sum.t() + row_sum) + TP ;
    uvec FP = sum(confMat, 0).t() - TP;
    uvec FN = sum(confMat, 1) - TP;

    /*mat binartConfMat = zeros<mat>(2, 2);
    binartConfMat(0,0) = TP(1);
    binartConfMat(0,1) = confMat(0,1);
    binartConfMat(1,0) = confMat(1,0);
    binartConfMat(1,1) = TP(0);*/

    return confMat;
}


void LogisticRegression::print_confusionMatrix(umat confMat) const
{
    unsigned int total_samples = accu(confMat);
    unsigned int total_TP = trace(confMat);

    uvec actuals = conv_to< uvec >::from(d_dset.labels());
    confMat.insert_cols(0, actuals);

    cout << endl << "confMat.size(): " << confMat.size() << endl;

    cout << endl << "       Confusion Matrix        "
         << endl << "     --------------------      "
         << endl << "          Predicted            "
         << endl << "        <----------->          "
         << endl << "                         ";

    for(unsigned int k=0; k<d_dset.K(); k++)
    {
        cout << k << "            ";
    }

    cout << endl << "                        ";
    for(unsigned int k=0; k<d_dset.K(); k++)
    {
        cout << "---          ";
    }

    cout << endl << confMat << endl;
    cout << endl << "Total Samples: " << total_samples
         << endl << "Total True Positives: " << total_TP;

    /*cout << endl << "No. of actual positives: " << TP + FN
         << endl << "No. of actual negatives: " << TN + FP
         << endl << "Total Misclassifications: " << FP + FN << endl;*/
}


double LogisticRegression::f1Score(const mat X, const mat labels, const bool show_stats=false) const
{
    umat confMat = confusionMatrix(X, labels);

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
