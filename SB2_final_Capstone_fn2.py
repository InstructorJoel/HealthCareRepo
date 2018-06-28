
# coding: utf-8

# In[17]:


#Function 2 (auto normal distribution plotter) - will automatically generate normal distribution 
#plots for all columns in an all-numeric dataframe that is passed to the function as an argument.  
#As a test case, I will provide a csv file containing potential measured values of total cholesterol 
#in a fictitious clinic for first 6 week days. User will need to read this in using pandas library, 
#store as a dataframe and pass the all numeric dataframe to the function. The goal is to plot 
#each day's "cholesterol" values into a normal distribution plot and compare day-t0-day distributions. 
#Significantly different daily measurements are easily spotted using this visual.
    
    
def auto_norm_dist_plotter(df):
    import pandas as pd # pandas will be needed for managing dataset
    import numpy as np # in case we need to use some of numpy functions
    from matplotlib import pyplot as plt # matplot lib import for additional customization
    get_ipython().run_line_magic('matplotlib', 'inline')
    import seaborn as sns
   
    ncol=df.columns.tolist() # saves column names with indices to a "list"
    print(str(ncol) + str("<-- column names in dataframe"))
    print("")

    means=[] # empty list created to store mean (average) of each numeric column in dataframe
    stds=[] # empty list created to store standard deviation of each numeric column in dataframe

    for i in range(len(ncol)):  # calculate and append mean and standard deviations of each colummn to their respective lists
        meanf=float(df[ncol[i]].mean())
        means.append(meanf)
        meansrnd = list(np.around(np.array(means),2))
        stdevf=float(df[ncol[i]].std())
        stds.append(stdevf)
        stdsrnd = list(np.around(np.array(stds),2))
        i=i+1

#display calculated mean and standard deviations for each numeric column in dataframe
    print("For columns in dataframe, calculated means (averages) - rounded to two digits - are: ")
    print(meansrnd)
    print("")
    print("For columns in dataframe, calculated standard deviations - rounded to two digits -  are: ")
    print(stdsrnd)
    print("")

    print("Based on calculated means and standard deviations, normal distribution plots for each column follow: ")

# display overlay plots of normal distribution curves from numeric column(s) in dataframe
    for i in range(len(means)):
        ax=sns.distplot(np.random.normal(means[i],stds[i],1000), hist=False, label = i+1)
        plt.xlabel("Values") # label X axis
        plt.ylabel("Probability") # label Y axis
        plt.legend(bbox_to_anchor=(1, 1), loc=2) # place legend to right
        i=i+1


# In[ ]:




