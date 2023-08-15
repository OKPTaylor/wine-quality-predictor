# -------------------------------------------------------------------IMPORTS-------------------------------------------------------------------
# Data viz:
import matplotlib.pyplot as plt

# Sklearn stuff:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt


# -------------------------------------------------------------------FUNCTIONS-------------------------------------------------------------------


def regression_errors(y, yhat):
        '''
        Returns the following values:
        root mean squared error (RMSE) and r-squared (R2)
        '''
        #import
        
        #calculate r2
        r2 = r2_score(y, yhat)
        #calculate MSE
        MSE = mean_squared_error(y, yhat)
        #calculate RMSE
        RMSE = sqrt(MSE)
        
        return RMSE, r2

# def test_poly_model(x_train_scaled, y_train):
    

def test_poly_model(y_train, y_test, x_train_scaled, x_test_scaled):
    # Polynomial regression
    # Make polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3)
    # Fit and transform x_train_scaled
    x_train_scaled_pf = pf.fit_transform(x_train_scaled)
    x_test_scaled_pf = pf.transform(x_test_scaled)  # Transform the test set as well

    # Fit the polynomial regression model
    pr = LinearRegression()
    pr.fit(x_train_scaled_pf, y_train)

    # Predict
    pred_pr_tr = pr.predict(x_train_scaled_pf)
    pred_pr_test = pr.predict(x_test_scaled_pf)

    # Evaluate polynomial regression
    rmse_train = mean_squared_error(y_train, pred_pr_tr, squared=False)
    r2_train = r2_score(y_train, pred_pr_tr)
    print("Train set - RMSE:", round((rmse_train),2), "R-squared:", round((r2_train),2))

    rmse_test = mean_squared_error(y_test, pred_pr_test, squared=False)
    r2_test = r2_score(y_test, pred_pr_test)
    print("Test set - RMSE:", round((rmse_test),2), "R-squared:", round((r2_test),2))
   
    #plot to visualize actual vs predicted using a histogram
    plt.figure(figsize=(16,8))

    plt.hist(y_train, color='blue', alpha=.5, label="Actual Wine Quality")
    plt.hist(pred_pr_test, color='red', alpha=.5, label="TEST Model: Polynomial Regression")
    plt.xlabel("Wine Quality")
    plt.ylabel("Number of Wines")
    plt.title("Comparing the Distribution of Wine Quality to Distributions of Predicted Wine Quality for the Test Model")
    plt.legend()
    plt.show()