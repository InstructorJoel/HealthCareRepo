
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
import matplotlib.pyplot as plt

#from datetime import datetime, date, time, timedelta

#Function for importing CSV File
def importcsv(var):
    imported = pd.read_csv(var)
    return imported

#Function to plot Histogram
def histogram(attribute,bins):
    plt.hist(attribute, bins)

df_list = list(df.columns[1:24])

#Correlation Plot Function
def corr_plot(list_of_df):
    corr = df[list_of_df].corr()
    #plotting the layout for map
    plt.figure(figsize=(25,25))
    sns.heatmap(corr, cmap='coolwarm', xticklabels = list_of_df,  yticklabels = list_of_df, annot=True)

#Function for Random Forest Classification
def RandomForest(dataset):
    X = dataset.iloc[:, [1, 23]].values
    y = dataset.iloc[:, 24].values

    # Splitting the dataset as Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Feature Scaling
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    # Classifying Random Forest to the Training set
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    classifier.fit(X_train, y_train)

    # Predicting results for test set
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    
    return conf_mat


if __name__ == "__main__":
  # Importing Data Frame
  df = importcsv("C:\\Users\\DTP\\Desktop\\cancer_info.csv")

  # Plotting Histogram
  histogram(df.Age, bins=5)

  #Plotting Correlation Plot
  corr_plot(df_list)
  
  #Applying Random forest function
  RandomForest(df)
