import scipy.stats as stats
import pandas as pd
import os
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn stuff:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import env




#This function returns a url for a mysql database based on the data base name and env credituals
def get_db_url(data_base):
    return (f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{data_base}')


#This function returns a dataframe based on the provided info
def new_sql_data_query(sql_query , url_for_query):
    return pd.read_sql(sql_query , url_for_query)
#Use this call with the above functions to pull back the data and save it: 
# dataframe_name = new_sql_data_query("desired_sql_query" , get_db_url("database_name"))
#This function will either load a csv file if it exist or run a sql query and save it to a csv
def get_sql_data(sql_query, url_for_query, filename):
    
    if os.path.exists(filename): 
        df = pd.read_csv(filename)
        print("csv found and loaded")
        return df
    else:
        df = new_sql_data_query(sql_query , url_for_query)

        df.to_csv(filename)
        return df
"""
You must have the following provided for the function to work:

sql_query = "desired query"
directory = os.getcwd()
url_for_query = acq.get_db_url("sql_database_name")
filename = "datbase_name.csv"

"""
#then use this call: dataframe_name = wrg.get_sql_data(sql_query, url_for_query, filename)

#fix ugly column names
def clean_col(df):
    new_col_name = []

    for col in df.columns:
        new_col_name.append(col.lower().replace('.', '_'))

    df.columns = new_col_name

    df.head()
    return df
#call should be: wrg.clean_col(df_name)


#looks for nulls and returns columns with nulls and columns
def is_it_null(df_name):
    for col in df_name.columns:
        print(f"{df_name[col].isna().value_counts()}") 
#call should be: wrg.is_it_null(df_name)  


#removes all columns with more than 5% nulls, NaNs, Na, Nones. This should only be used when it makes sense to drop the column; may need to drop rows
def null_remove(df_name):
    for col in df_name.columns:
        if df_name[col].isna().value_counts("False")[0] < 0.95: #tests if a row cotains more than 5% nulls, NaNs, ect. 
            df_name.drop(columns=[col], inplace=True)
            print(f"Column {col} has been dropped because it contains more than 5% nulls")   
#call should be: wrg.null_remove(df_name)           

#brings back all the columns that may be duplicates of other columns
def col_dup(df_name):
    for col1 in df_name.columns:
        for col in df_name.columns:
            temp_crosstab=pd.crosstab(df_name[col] , df_name[col1])
            if temp_crosstab.iloc[0,0] != 0 and temp_crosstab.iloc[0,1] == 0 and temp_crosstab.iloc[1,0] == 0 and temp_crosstab.iloc[1,1] !=0:
                if col1 != col:
                    print(f"\n{col1} and {col} may be the duplicates\n")
                    
                    print(temp_crosstab.iloc[0:3,0:3])
                    print("--------------------------------------------")
       
#call should look like: wrg.col_dup(df_name)  

#the two following functions is for ploting univar
#makes a list of all var
def df_column_name(df_name):
    col_name = []
    for x in df_name.columns[0:]: #check to make sure the range is what you want
        col_name.append(x)   
    return col_name       

def plot_uni_var(df_name):    #plots univar
    for col in (df_column_name(df_name)):
        plt.hist(df_name[col])
        plt.title(col)
        plt.show()         
          
#splits your data into train, validate, and test sets for cat target var
def split_function_cat_target(df_name, target_varible_column_name):
    train, test = train_test_split(df_name,
                                   random_state=123, #can be whatever you want
                                   test_size=.20,
                                   stratify= df_name[target_varible_column_name])
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25,
                                   stratify= train[target_varible_column_name])
    return train, validate, test
#call should look like: 
#train_df_name, validate_df_name, test_df_name = wrg.split_function_cat_target(df_name, 'target_varible_column_name')

#splits your data into train, validate, and test sets for cont target var
def split_function_cont_target(df_name):
    train, test = train_test_split(df_name,
                                   random_state=123, #can be whatever you want
                                   test_size=.20)
                                   
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25)
    return train, validate, test
#call should look like: 
#train_df_name, validate_df_name, test_df_name = wrg.split_function_cont_target(df_name)

#This makes two lists containing all the categorical and continuous variables
def cat_and_num_lists(df_train_name):
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numeric varibles

    for col in df_train_name.columns[1:21]: #make sure to set this to the range you want
        
        if df_train_name[col].dtype == 'O':
            col_cat.append(col)
        else:
            if len(df_train_name[col].unique()) < 4: #making anything with less than 4 unique values a catergorical value
                col_cat.append(col)
            else:
                col_num.append(col)
    print(f"The categorical variables are: \n {col_cat} \n") 
    print(f"The continuous variables are: \n {col_num} \n")                
    return col_cat , col_num           
#the call for this should be: wrg.cat_and_num_lists(df_train_name)

#plots all pairwise relationships along with the regression line for each col cat and col num pair
def plot_variable_target_pairs(df_train_name,target_var):

    #df_train_name = df_train_name.sample(100000, random_state=123) #this is for sampling the data frame. This may not be needed for your data set
    col_cat, col_num = cat_and_num_lists(df_train_name)
    
    for col in col_num:
        print(f"{col.upper()} and {target_var}")
        
        sns.lmplot(data=df_train_name, x=col, y=target_var,
          line_kws={'color':'red'})
        plt.show()

#This plots all categorical variables against the target variable
def plot_categorical_and_target_var(df_train_name, target, cat_count=4): #this defaults to 4 unique values
    col_cat, col_num = cat_and_num_lists(df_train_name, cat_count)
    for col in col_cat:
        sns.barplot(x=df_train_name[col], y=df_train_name[target])
        plt.title(f"{col.lower().replace('_',' ')} vs {target}")
        plt.show()
        
        print()
        
#plots pairplots 
def pairplot_everything(df_train_name, cat):
    #df_train_name = df_train_name.sample(10000, random_state=123) #this is for sampling the data frame
    sns.pairplot(data=df_train_name, corner=True, hue=cat)

def corr_heatmap(df_train_name):
    #df_train_name = df_train_name.sample(10000, random_state=123) #this is for sampling the data frame
    plt.figure(figsize=(12,10))
    sns.heatmap(df_train_name.corr(), cmap='Blues', annot=True, linewidth=0.5, mask= np.triu(df_train_name.corr())) 
    plt.show()   


#This function is for running through catagorical on catagorical features graphing and running the chi2 test on them (by Alexia)
def cat_on_cat_graph_loop(dataframe_train_name, col_cat, target_ver, target_ver_column_name):
    for col in col_cat:
        print()
        print(col.upper())
        print(dataframe_train_name[col].value_counts())
        print(dataframe_train_name[col].value_counts(normalize=True))
        dataframe_train_name[col].value_counts().plot.bar()
        plt.show()
        print()
        print()
        print(f'HYPOTHESIZE')
        print(f"H_0: {col.lower().replace('_',' ')} does not affect {target_ver}")
        print(f"H_a: {col.lower().replace('_',' ')} affects {target_ver}]")
        print()
        print(f'VISUALIZE')
        sns.barplot(x=dataframe_train_name[col], y=dataframe_train_name[target_ver_column_name])
        plt.title(f"{col.lower().replace('_',' ')} vs {target_ver}")
        plt.show()
        print()
        print('ANALYZE and SUMMARIZE')
        observed = pd.crosstab(dataframe_train_name[col], dataframe_train_name[target_ver_column_name])
        chi2Test(observed)
        print()
        print()
#the call should be: prep.cat_on_cat_graph_loop(dataframe_train_name, col_cat, "target_ver", "target_ver_column_name")        

#this funciton works in this module to run the chi2 test with the above function
def chi2Test(observed):
    alpha = 0.05
    chi2, pval, degf, expected = stats.chi2_contingency(observed)
    print('Observed')
    print(observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p-value = {pval:.4f}')
    print('----')
    if pval < alpha:
        print ('We reject the null hypothesis.')
    else:
        print ("We fail to reject the null hypothesis.")
# prep.chi2Test(observed) is the call 

#a function that tests for normality of target variable using the Shapiro-Wilk test
def normality_test(df_train_name, target_var):
    alpha = 0.05
    stat, p = stats.shapiro(df_train_name[target_var])
    print(f'Statistics={stat}, p={p}' )
    if p > alpha:
        print('The data is normally distributed (fail to reject H0)')
    else:
        print('The sample is NOT normally distributed (reject H0)')


#This funcition runs through the continuous varaibles and the continuous target variable and runs the pearsonr test on them
def pearsonr_loop(df_train_name, target_var, cat_count=4):
    alpha = 0.05
    col_cat, col_num = cat_and_num_lists(df_train_name, cat_count)
    for col in col_num:
        sns.regplot(x=df_train_name[col], y=df_train_name[target_var], data=df_train_name, line_kws={"color": "red"})
        plt.title(f"{col.lower().replace('_',' ')} vs {target_var}")
        plt.show()
        print(f"{col.upper()} and {target_var}")
        corr, p = stats.pearsonr(df_train_name[col], df_train_name[target_var])
        print(f'corr = {corr}')
        print(f'p = {p}')
        if p < alpha:
            print('We reject the null hypothesis, there is a linear relationship between the variables\n')
        else:
            print('We fail to reject the null hypothesis, there is not a linear relationship between the variables\n') 
        


#This function runs through the continuous variables and the continuous target variable and runs the spearman test on them
def spearman_loop(df_train_name, target_var, cat_count=4):
    alpha = 0.05
    col_cat, col_num = cat_and_num_lists(df_train_name, cat_count)
    for col in col_num:
        sns.regplot(x=df_train_name[col], y=df_train_name[target_var], data=df_train_name, line_kws={"color": "red"})
        plt.title(f"{col.lower().replace('_',' ')} vs {target_var}")
        plt.show()
        print(f"{col.upper()} and {target_var}")
        corr, p = stats.spearmanr(df_train_name[col], df_train_name[target_var])
        print(f'corr = {corr}')
        print(f'p = {p}')
        if p < alpha:
            print('We reject the null hypothesis, there is a linear relationship between the variables\n')
        else:
            print('We fail to reject the null hypothesis, there is not a linear relationship between the variables\n')    


#This function takes the data and scales it using the MinMaxScaler
def scale_data(train, validate, test):
    to_scale=train.columns.tolist()

    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #this scales stuff 
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled 
# call should be train_scaled, validate_scaled, test_scaled = wrg.scale_data(x_train, x_validate, x_test)

#find and load csv file as a dataframe
def get_csv(csv_name):
    df = pd.read_csv(csv_name)
    return df




