# My Modules
import stats_conclude as sc

# Imports
import os

# Numbers
import pandas as pd 
import numpy as np
from scipy import stats

# Vizzes
import matplotlib.pyplot as plt
import seaborn as sns

# Splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------

print(f"Load in successful... awaiting commands")

# ACQUIRE

def get_csv(csv_name):
    df = pd.read_csv(csv_name)
    print(f"CSV found")
    return df


# -------------------------------------------------------------------------

#  EXPLORE

def wrangle_wine_initial():
    """This function takes in two local CSV files, combines them, performs data cleanup
    and returns a clean df.
    ---
    This function does not remove outliers.
    """
    print(f"Acquire Details")
    
    # get red wine CSV
    red_wine = get_csv("winequality-red.csv")
    
    # get white wine CSV
    white_wine = get_csv("winequality-white.csv")
    
    # make new column
    red_wine['color'] = 'red'
    white_wine['color'] = 'white'
    
    # combine into one df
    df = pd.concat([red_wine, white_wine])
    print(f"Combined CSV's into one DF")
    
    # rename columns
    df_clean = df.rename(columns={'fixed acidity':'fixed_acidity',
                         'volatile acidity':'volatile_acidity',
                         'citric acid':'citric_acid',
                         'residual sugar':'residual_sugar',
                         'chlorides':'sodium',
                         'free sulfur dioxide':'free_SO2_shelf_life',
                         'total sulfur dioxide':'total_SO2_processed_level',
                         'density':'density',
                         'pH':'pH',
                         'sulphates':'preservatives',
                         'alcohol':'alcohol',
                         'quality':'quality',
                         'color':'type'})
    print(f"--------------------------------------------")
    print(f"Prepare Details\nRenamed columns for ease of use")
    
    # encode the categorical column
    dummy_df = pd.get_dummies(df_clean[['type']], dummy_na=False, drop_first=[True])
    df_clean = pd.concat([df_clean, dummy_df], axis=1)
    df_clean = df_clean.drop(columns=('type'))
    print(f"Encoded Type column")

    return df_clean

def outliers(df):
    """This function takes in a df and identifies outliers for each numerical column
    """
    # empty lists
    col_cat = [] #this is for my categorical variables 
    col_num = [] #this is for my numerical variables 
    
    # iterate
    for col in df.columns: 
        if col in df.select_dtypes(include=['int64', 'float64']): 
            col_num.append(col) 
        else: 
            col_cat.append(col) 

    for col in col_cat: 
        print(f"{col.capitalize().replace('_', ' ')} is a categorical column.") 
    print(f"--------------------------------------------")
    print('Outliers Calculated with IQR Ranges, multiplier 1.5')
    print(f"--------------------------------------------")
    for col in col_num: 
        q1 = df[col].quantile(.25) 
        q3 = df[col].quantile(.75) 
        iqr = q3 - q1 
        upper_bound = q3 + (1.5 * iqr) 
        lower_bound = q1 - (1.5 * iqr) 
        print(f"{col.capitalize().replace('_', ' ')} < = {upper_bound.round(2)} and > {lower_bound.round(2)}") 

def wrangle_wine_extra():
    """This function takes in two local CSV files, combines them, performs data cleanup
    and returns a clean df.
    ---
    This function does remove outliers as well as prints statements as to the 
    exact outliers removed. 
    """
    print(f"Prepare Details")
    
    # get red wine CSV
    red_wine = get_csv("winequality-red.csv")
    
    # get white wine CSV
    white_wine = get_csv("winequality-white.csv")
    
    # make new column
    red_wine['color'] = 'red'
    white_wine['color'] = 'white'
    
    # combine into one df
    df = pd.concat([red_wine, white_wine])
    print(f"Combined CSV's into one DF")
    
    # rename columns
    df_clean = df.rename(columns={'fixed acidity':'fixed_acidity',
                         'volatile acidity':'volatile_acidity',
                         'citric acid':'citric_acid',
                         'residual sugar':'residual_sugar',
                         'chlorides':'sodium',
                         'free sulfur dioxide':'free_SO2_shelf_life',
                         'total sulfur dioxide':'total_SO2_processed_level',
                         'density':'density',
                         'pH':'pH',
                         'sulphates':'preservatives',
                         'alcohol':'alcohol',
                         'quality':'quality',
                         'color':'type'})
    print(f"--------------------------------------------")
    print(f"Renamed columns for ease of use")

    # outliers
    col_cat = [] #this is for my categorical variables 
    col_num = [] #this is for my numerical variables 
    for col in df_clean.columns: 
        if col in df_clean.select_dtypes(include=['int64', 'float64']): 
            col_num.append(col) 
        else: 
            col_cat.append(col) 

    for col in col_cat: 
        print(f"{col.capitalize().replace('_', ' ')} is a categorical column.") 
    print(f"--------------------------------------------")
    print('Outliers Calculated with IQR Ranges, multiplier 1.5')

    for col in col_num: 
        q1 = df_clean[col].quantile(.25) 
        q3 = df_clean[col].quantile(.75) 
        iqr = q3 - q1 
        upper_bound = q3 + (1.5 * iqr) 
        lower_bound = q1 - (1.5 * iqr) 
        print(f"{col.capitalize().replace('_', ' ')} < = {upper_bound.round(2)} and > {lower_bound.round(2)}") 
        df_clean = df_clean[(df_clean[col] <= upper_bound) & (df_clean[col] >= lower_bound)] 

    print(f"Outliers removed: Percent Original Data Remaining: {round(df_clean.shape[0]/df.shape[0]*100,0)}")
    
    # encode the categorical column
    dummy_df = pd.get_dummies(df_clean[['type']], dummy_na=False, drop_first=[True])
    df_clean = pd.concat([df_clean, dummy_df], axis=1)
    print(f"Encoded Type column")

    return df_clean

# -------------------------------------------------------------------------

# SPLIT

def split_data(df):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.
    ---
    Format: train, validate, test = function()
    '''
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25,
                                       random_state=123)
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

def split_data_stratify(df, target):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.
    ---
    Format: train, validate, test = function()
    '''
    train, test = train_test_split(df, test_size=.2,
                                        random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25,
                                       random_state=123, stratify=train[target])
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

# X_train, y_train, X_validate, y_validate, X_test, y_test WITH TARGET

def x_y_train_validate_test_stratify(train, validate, test, target):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    ---
    Format: X_train, y_train, X_validate, y_validate, X_test, y_test = function()
    """ 
    # X_train, validate, and test to be used for modeling
    X_train = train.drop(columns=[target])
    y_train = train[{target}]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[{target}]
   
    X_test = test.drop(columns=[target])
    y_test = test[{target}]

    print(f"Variable assignment successful...")

    # verifying number of features and target
    print(f"Verifying number of features and target:")
    print(f'Train: {X_train.shape, y_train.shape}')
    print(f'Validate: {X_validate.shape, y_validate.shape}')
    print(f'Test: {X_test.shape, y_test.shape}')

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# X_train, y_train, X_validate, y_validate, X_test, y_test WITH TARGET

def x_y_train_validate_test_no_stratify(train, validate, test):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    ---
    Format: X_train, X_validate, X_test = function()
    """ 
    # X_train, validate, and test to be used for modeling
    X_train = train

    X_validate = validate
   
    X_test = test

    print(f"Variable assignment successful...")

    # verifying number of features and target
    print(f"Verifying number of features and target:")
    print(f'Train: {X_train.shape}')
    print(f'Validate: {X_validate.shape}')
    print(f'Test: {X_test.shape}')

    return X_train, X_validate, X_test


# -------------------------------------------------------------------------

# SCALING
   
def scaler_minmax(train, validate, test):
    """This function takes in the train, validate, and test datasets,
    splits using MinMaxScaler, random seed=123, and returns 3 scaled df's
    ---
    Format: X_train_scaled, X_validate_scaled, X_test_scaled = function()
    """
    #to_scale
    to_scale = train.columns.tolist()
    
    #make copies for scaling
    X_train_scaled = train.copy()
    X_validate_scaled = validate.copy()
    X_test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    X_train_scaled[to_scale] = scaler.transform(train[to_scale])
    X_validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    X_test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

def scaler_cluster(df):
    """This function takes in the train dataset, splits using MinMaxScaler, 
    random seed=123, and returns 1 scaled df
    ---
    Format: X_train_cluster_scaled = function()
    """
    #to_scale
    to_scale = df.columns.tolist()
    
    #make copies for scaling
    X_train_cluster_scaled = df.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(df[to_scale])

    #use the thing
    X_train_cluster_scaled[to_scale] = scaler.transform(df[to_scale])
    
    return X_train_cluster_scaled

def inverse_minmax(scaled_df):
    """This function takes in the MinMaxScaler object and returns the inverse
    of a single scaled df input.
    
    format to return original df = minmaxscaler_back = function()
    """
    from sklearn.preprocessing import MinMaxScaler
    minmaxscaler_inverse = pd.DataFrame(MinMaxScaler.inverse_transform(scaled_df))

    # visualize if you want it too
    # plt.figure(figsize=(13, 6))
    # plt.subplot(121)
    # plt.hist(X_train_scaled_ro, bins=50, ec='black')
    # plt.title('Scaled')
    # plt.subplot(122)
    # plt.hist(robustscaler_back, bins=50, ec='black')
    # plt.title('Inverse')
    # plt.show()

    return minmaxscaler_inverse