
#Author Akshay Kapoor 

#Please refer the AkshayBreastCancerAnalysis.ipynb notebook for better explanation, I have made this file as there were problems in importing functions from .ipynb notebook. So file is just made for re usability  
# coding: utf-8

# # Breast cancer detection
# dataset source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 

# We have 30 different attributes from images extracted, Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. <br>
# We predict the Stage of Breast Cancer B (Bengin) or M (malignant).

# **Attribute Information:**
# <br>
# 1) ID number <br>
# 2) Diagnosis (M = malignant, B = benign) <br>
# 3-32) <br>
# 
# Ten real-valued features are computed for each cell nucleus: <br>
# 
# a) radius (mean of distances from center to points on the perimeter) <br>
# b) texture (standard deviation of gray-scale values) <br>
# c) perimeter <br>
# d) area <br>
# e) smoothness (local variation in radius lengths) <br>
# f) compactness (perimeter^2 / area - 1.0) <br>
# g) concavity (severity of concave portions of the contour) <br>
# h) concave points (number of concave portions of the contour) <br>
# i) symmetry <br>
# j) fractal dimension ("coastline approximation" - 1) <br>

# In[3]:


#Importing the libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def data_analysis():
    df = pd.read_csv("AkshayData.csv")
    df.head()
    df.info()
    df.drop('Unnamed: 32', axis  = 1, inplace=True)
    df.drop('id', axis = 1, inplace= True)
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
    df.describe()
    sns.countplot(df['diagnosis'])
    df.columns
    first = list(df.columns[1:10])
    train, test = train_test_split(df, test_size = 0.25)
  
    second = list(df.columns[11:21])
    third =  list(df.columns[21:30])
    



    corr1 = df[first].corr()
    #Let us visualize with a heatmap
    plt.figure(figsize=(14,10))
    sns.heatmap(corr1, cmap='coolwarm', xticklabels = first,  yticklabels = first, annot=True)


    # **We can see that radius, perimeter and area are highly correlated as seen from the heatmap.** <br>
    # **Also compactness_mean, concavepoint_mean and concavity_mean are highly correlated**

    #Let us perform analysis on the mean features

    melign = df[df['diagnosis'] == 1][first]
    bening = df[df['diagnosis'] == 0][first]

    melign.columns

    for columns in melign.columns:
        plt.figure()
        sns.distplot(melign[columns], kde=False, rug= True)
        sns.distplot(bening[columns], kde=False, rug= True)
        sns.distplot
        plt.tight_layout()

        

# We can see that the mean values of perimeter, area, concavity, compactness, radius and concave points can be used for classification as these parameters show a correlation. <br>
# While parameters such as smoothness, symmetry, fractual dimension and texture don't show much seperation and is of not much use for classification.
    color_function = {0: "green", 1: "red"}
    colors = df["diagnosis"].map(lambda x: color_function.get(x))

    pd.plotting.scatter_matrix(df[first], c=colors, alpha = 0.4, figsize = (15, 15))
  #
  
    # **Using a scatter matrix we can see a well seperation of malign and benign cancer with green points indication benign cancer cells and red points indicating malign cancer cells.**
 
    #We divide the data into Training and test set 
    


# I have created a function to perform k folds cross validation which helps in obtaining a better insight to test the accuracy of the model
# More info at https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0],n_folds= 5)
  error = []
  for train, test in kf:
    # Filter the training data
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  model.fit(data[predictors],data[outcome]) 


def mac_classification():

#Using Logistic regression on the top five features
#more info at https://en.wikipedia.org/wiki/Logistic_regression
    df = pd.read_csv("AkshayData.csv")
    df.drop('Unnamed: 32', axis  = 1, inplace=True)
    df.drop('id', axis = 1, inplace= True)
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
    first = list(df.columns[1:10])
    train, test = train_test_split(df, test_size = 0.25)
  
    
    predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
    outcome_var='diagnosis'
    model=LogisticRegression()
    classification_model(model,train,predictor_var,outcome_var)

#Let us check the accuracy on test data
    classification_model(model, test,predictor_var,outcome_var)

#Let us try to classify using a decision tree classifier 
    predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
    model = DecisionTreeClassifier()
    classification_model(model,train,predictor_var,outcome_var)


# We are getting 100% accuracy! Is it overfitting let us try it on test data
    classification_model(model, test,predictor_var,outcome_var)
# Let us try using random forest

    predictor_var = first
    model = RandomForestClassifier()
    classification_model(model, train,predictor_var,outcome_var)

#Let us find the most important features used for classification model

    featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    print(featimp)
    predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean']
    model = RandomForestClassifier()
    classification_model(model,train,predictor_var,outcome_var)

# I think we get a better prediction with all the features now let us try it on test data!
    predictor_var = first
    model = RandomForestClassifier()
    classification_model(model, test,predictor_var,outcome_var)

if __name__ == "__main__":
    data_analysis()
    mac_classification()

# ## Conclusion
# 
# Hence we can see detailed exploratory data analysis of breast cancer data and implementation of classification algorithms to train a model in detecting whether the cancer is benign or malign.
# 
