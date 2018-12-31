# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 01:28:54 2018

@author: tdpco
"""
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics

# importing the dataset
dataset = pd.read_csv('smsspamcollection.tsv', sep='\t')

# Viewing the dataset
print("Printing First Five Rows of Dataset: \n")
print(dataset.head())

# Dividing the dataset into dependent and independent variable
X = dataset[['length', 'punct']]
y = dataset['label']

# Splitting the dataset into test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Creating a function to check for model
def choose_model(X_train, y_train, X_test, model_name):
    """ Choose among the three models """
    if (model_name == "Logistic Regression"):
        model = LogisticRegression(solver='lbfgs')
    elif (model_name == "Naive Bayes"):
        model = MultinomialNB()
    elif (model_name == "SVM"):
        model = SVC(gamma='auto')
    else:
        print("No Correct Model Choosen")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Creating a function to print result
def print_result(X_train,y_train,X_test,y_test,model_name):
    # Printing some relevant result
    predictions = choose_model(X_train, y_train, X_test, model_name)
    result = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['Ham', 'Spam'], columns = ['Ham', 'Spam'])
    print('\n\n')
    print(result)
    print('\n\n')
    print(metrics.classification_report(y_test,predictions))
    print('\n\n')
    accuracy = metrics.accuracy_score(y_test,predictions)
    print(f"Accuracy of {model_name} is {accuracy}")

# Asking user to choose model
print("\n\n\t\tHello User, Please Choose Model to run: ")
print("\t\t\t 1.Logistic Regression \n\t\t\t 2.Naive_Bayes\n\t\t\t 3.SVM\n\t\t\t 4.Run All")
user_input = input("Enter your choice: ")
user_choice = int(user_input)
if user_choice == 1:
    model_name = "Logistic Regression"
    print_result(X_train,y_train,X_test,y_test,model_name)
elif user_choice == 2:
    model_name = "Naive Bayes"
    print_result(X_train,y_train,X_test,y_test,model_name)
elif user_choice == 3:
    model_name == "SVM"
elif user_choice == 4:
    model_names = ["Logistic Regression","Naive Bayes","SVM"]
    for model in model_names:
        predictions = choose_model(X_train, y_train, X_test, model)
        accuracy = round(metrics.accuracy_score(y_test,predictions) * 100,2)
        print(f"Accuracy of {model} is {accuracy} %")
else:
    print("Sorry Wrong Choice Selected")
