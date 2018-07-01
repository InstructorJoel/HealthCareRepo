
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
get_ipython().run_line_magic('matplotlib', 'inline')

#scikit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# ## Let us learn more about the data 

# In[4]:


df = pd.read_csv("AkshayData.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


#We can see Unnamed:32 has all null values hence we cannot use this column for our analysis and id will also be of no use for analysis
df.drop('Unnamed: 32', axis  = 1, inplace=True)
df.drop('id', axis = 1, inplace= True)


# In[8]:


#Let us convert 'Malign' and 'Benign' to 1 and 0 respectively so it will be easier for analysis

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})


# In[9]:


df.describe()


# ## Performing Exploratory Data Analysis
# 

# In[10]:


sns.countplot(df['diagnosis'])


# We can see there are almost double number patients with benign cancer

# In[11]:


df.columns


# In[12]:


#The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image,resulting in 30 features.
#For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
#more info at https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

first = list(df.columns[1:10])
second = list(df.columns[11:21])
third =  list(df.columns[21:30])


# In[13]:


#Let us find the correlation between different attributes
corr1 = df[first].corr()


# In[14]:


#Let us visualize with a heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr1, cmap='coolwarm', xticklabels = first,  yticklabels = first, annot=True)


# **We can see that radius, perimeter and area are highly correlated as seen from the heatmap.** <br>
# **Also compactness_mean, concavepoint_mean and concavity_mean are highly correlated**
# 

# In[15]:


#Let us perform analysis on the mean features

melign = df[df['diagnosis'] == 1][first]
bening = df[df['diagnosis'] == 0][first]


# In[16]:


melign.columns


# In[17]:


for columns in melign.columns:
    plt.figure()
    sns.distplot(melign[columns], kde=False, rug= True)
    sns.distplot(bening[columns], kde=False, rug= True)
    sns.distplot
plt.tight_layout()


# We can see that the mean values of perimeter, area, concavity, compactness, radius and concave points can be used for classification as these parameters show a correlation. <br>
# While parameters such as smoothness, symmetry, fractual dimension and texture don't show much seperation and is of not much use for classification.
# 

# In[18]:


color_function = {0: "green", 1: "red"}
colors = df["diagnosis"].map(lambda x: color_function.get(x))

pd.plotting.scatter_matrix(df[first], c=colors, alpha = 0.4, figsize = (15, 15));


# **Using a scatter matrix we can see a well seperation of malign and benign cancer with green points indication benign cancer cells and red points indicating malign cancer cells.**
# 

# ## Machine learning
# 

# In[19]:


#We divide the data into Training and test set 
train, test = train_test_split(df, test_size = 0.25)


# In[20]:


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


# In[21]:


#Using Logistic regression on the top five features
#more info at https://en.wikipedia.org/wiki/Logistic_regression

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var='diagnosis'
model=LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)


# In[22]:


#Let us check the accuracy on test data
classification_model(model, test,predictor_var,outcome_var)


# In[23]:


#Let us try to classify using a decision tree classifier 
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)


# We are getting 100% accuracy! Is it overfitting let us try it on test data
# 

# In[24]:


classification_model(model, test,predictor_var,outcome_var)


# Let us try using random forest

# In[25]:


predictor_var = first
model = RandomForestClassifier()
classification_model(model, train,predictor_var,outcome_var)


# In[26]:


#Let us find the most important features used for classification model

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)


# In[27]:


predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)


# In[28]:


# I think we get a better prediction with all the features now let us try it on test data!
predictor_var = first
model = RandomForestClassifier()
classification_model(model, test,predictor_var,outcome_var)


# ## Conclusion
# 
# Hence we can see detailed exploratory data analysis of breast cancer data and implementation of classification algorithms to train a model in detecting whether the cancer is benign or malign.
# 
