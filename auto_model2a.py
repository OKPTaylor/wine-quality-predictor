import pandas as pd
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn stuff:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#call should be: atm.auto_model_knn(X_train, y_train, X_validate, y_validate, k_range)        

def auto_tree(x_train, y_train, x_validate, y_validate, max_depth_range=(5,10,13,15,20)):
    scores_all = []

    for x in max_depth_range:

        tree = DecisionTreeClassifier(max_depth=x) #creates, fits, scores train data
        tree.fit(x_train, y_train)
        train_acc = tree.score(x_train, y_train)
        y_pred = tree.predict(x_train)

        #conf = confusion_matrix(y_train, y_pred)
        
       
        print(f"\n------------------------ Train Model with depth of {x} ------------------------------")
        #plot_confusion_matrix(tree, x_train, y_train)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 
        
        #evaluate on my validate data
        val_acc = tree.score(x_validate, y_validate)
        y_vpred = tree.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        print(f"\n---------------------Validate Model with depth of {x}---------------------------------")
        #plot_confusion_matrix(tree, x_validate, y_validate)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_vpred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf))
        print(f'\nk number = {x} accuracy = {train_acc, val_acc}')

        scores_all.append([x, train_acc, val_acc])

def auto_random_trees(x_train, y_train, x_validate, y_validate, max_depth=[5,8,10,13,15,18,20]):
    scores_all = []

    for x in max_depth:
        
        #make it
        rf = RandomForestClassifier(random_state=123, max_depth=x) #increases the sample leaf whild decreasing max depth
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)
        y_pred = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
    
        print(f"\n------------------------ Train Model with depth of {x} ------------------------------")
        #plot_confusion_matrix(rf, x_train, y_train)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 
        
        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)
        y_vpred = rf.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        print(f"\n---------------------Validate Model with depth of {x}---------------------------------")
        #plot_confusion_matrix(rf, x_validate, y_validate)
        #plt.show()
        print(pd.DataFrame(classification_report(y_validate, y_vpred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf))
        print(f'\ndepth = {x} accuracy = {train_acc, val_acc}')

        scores_all.append([x, train_acc, val_acc])

def auto_lo_regress(x_train, y_train, x_validate, y_validate, max_depth=[5,8,10,13,15,18,20]):
        scores_all = []

        for x in max_depth:
        
        #make it
            logit = LogisticRegression() #increases the sample leaf whild decreasing max depth
                #fit it
            logit.fit(x_train, y_train)
                #transform it
            train_acc = logit.score(x_train, y_train)
            y_pred = logit.predict(x_train)
            #conf = confusion_matrix(y_train, y_pred)
            
            print(f"\n------------------------ Train Model with C of {x} ------------------------------")
            #plot_confusion_matrix(logit, x_train, y_train)
            #plt.show()
            print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
            print("------------ Metrics ----------")
            #print(print_cm_metrics(conf)) 
                
                #evaluate on my validate data
            val_acc = logit.score(x_validate, y_validate)
            y_vpred = logit.predict(x_validate)
            #conf = confusion_matrix(y_validate, y_vpred)

            print(f"\n---------------------Validate Model with C of {x}---------------------------------")
            #plot_confusion_matrix(logit, x_validate, y_validate)
            #plt.show()
            print(pd.DataFrame(classification_report(y_validate, y_vpred, output_dict=True)))
            print("------------ Metrics ----------")
            #print(print_cm_metrics(conf))
            print(f'\nk number = {x} accuracy = {train_acc, val_acc}')

            scores_all.append([x, train_acc, val_acc])



'''----------------------------------------------- Auto Modeling No Graphs ---------------------------------------------------------------------------------'''

#This function runs through a range of k values and returns the accuracy scores for each k value       
def auto_knn_scores(X_train, y_train, X_validate, y_validate, k_range=[3,5,7,8,12,18]): # k_range default should be set to 5
    scores_all = []
    

    for x in k_range:
        
        #make it
        knnx = KNeighborsClassifier(n_neighbors=x) #increases the k number
        #fit it
        knnx.fit(X_train, y_train)
        #transform it
        train_acc = knnx.score(X_train, y_train)
        y_pred = knnx.predict(X_train)
        
        conf = confusion_matrix(y_train, y_pred)
        
        #evaluate on my validate data
        val_acc = knnx.score(X_validate, y_validate)
        y_vpred = knnx.predict(X_validate)
        conf = confusion_matrix(y_validate, y_vpred)
        
        scores_all.append([x, train_acc, val_acc])
    #create a dataframe from the scores_all list of lists
    df_scores = pd.DataFrame(scores_all)
    #add column names to the dataframe
    df_scores.columns = ['k', 'train_accuracy', 'validate_accuracy']
    # sort the dataframe by the difference between train and validate
    df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
    print(df_scores.sort_values(by='delta', ascending=False))

#call should be: atm.auto_knn_no_graph(X_train, y_train, X_validate, y_validate, k_range) 

#This function runs through a range of max_depth values and returns the accuracy scores for each max_depth value
def auto_random_forest_scores(x_train, y_train, x_validate, y_validate, max_depth=[5,8,10,13,15,18,20]):
    scores_all = []

    for x in max_depth:
        
        #make it
        rf = RandomForestClassifier(random_state=123, max_depth=x) #increases the sample leaf whild decreasing max depth
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)
        y_pred = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
    
        
        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)
        y_vpred = rf.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        scores_all.append([x, train_acc, val_acc]) 

    #create a dataframe from the scores_all list of lists
    df_scores = pd.DataFrame(scores_all)
    # add column names to the dataframe
    df_scores.columns = ['max_depth', 'train_accuracy', 'validate_accuracy']
    # sort the dataframe by the difference between train and validate
    df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
    print(df_scores.sort_values(by='delta', ascending=False))

#This function runs through a range of max_depth values and returns the accuracy scores for each max_depth value for logistic regression
def auto_lo_regress_scores(x_train, y_train, x_validate, y_validate, max_depth=[5,8,10,13,15,18,20]):
        scores_all = []

        for x in max_depth:
        
        #make it
            logit = LogisticRegression() #increases the sample leaf whild decreasing max depth
                #fit it
            logit.fit(x_train, y_train)
                #transform it
            train_acc = logit.score(x_train, y_train)
            y_pred = logit.predict(x_train)
            #conf = confusion_matrix(y_train, y_pred) 
                
            #evaluate on my validate data
            val_acc = logit.score(x_validate, y_validate)
            y_vpred = logit.predict(x_validate)
            #conf = confusion_matrix(y_validate, y_vpred)

            

            scores_all.append([x, train_acc, val_acc])
        #create a dataframe from the scores_all list of lists
        df_scores = pd.DataFrame(scores_all)
        # add column names to the dataframe
        df_scores.columns = ['max_depth', 'train_accuracy', 'validate_accuracy']
        # sort the dataframe by the difference between train and validate
        df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
        print(df_scores.sort_values(by='delta', ascending=False))        
