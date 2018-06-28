
# coding: utf-8

# In[ ]:


##To Function 1 (manual normal distribution plotter ) - will allow users to enter up to 5 mean and standard distribution 
##values and the function will generate a normal distribution curve given these user entries.  The user will also be 
## able to view entered values of mean and standard deviation to ensure there is no entry error on user.  For testing 
## this function, I will provide annual weather averages along with their standard deviations from a limited dataset 
##for Victoria, Canada.  Function 1 is limited to only 5 manual entries as this can be error-prone process.
##Consider using auto_normal_dist_plotter in when you have a dataframe with numeric columns instead of this function


def manual_norm_dist_plotter():
    from matplotlib import pyplot as plt # matplot lib import for additional customization
    get_ipython().run_line_magic('matplotlib', 'inline')
    import pandas as pd
    import numpy as np
    import seaborn as sns # use seaborn for plotting
    
    # create an empty list to store up to 5 "mean (i.e., average)" measurements/values from up to 5 "trials"
    means = [] 
    maxLenMeans = 5
    while len(means) < maxLenMeans:
        meanval = input("Enter mean values from up to 5 trials (or nothing to skip): ")
        if meanval == '':
            break
        meanvaluef=float(meanval)
        means.append(meanvaluef)

    print(str(means) + " <- Check mean values entered. If incorrect, skip next step and restart") # allows user to check if entries are correct
    print("")

    # create an empty list to store up to 5 user entered values for "standard deviations" from each "trial" used above for entering means
    stdev=[]
    maxlenstdev = 5
    while len(stdev)<maxlenstdev:
        stdevval=input("Enter up to 5 standard deviation values (from) same samples used for means (or nothing to skip): ")
        if stdevval == '':
            break
        stdevf=float(stdevval)
        stdev.append(stdevf)
    
    print(str(stdev) + " <-Check standard deviation values entered. If incorrect, restart from beginning")
    print("Unless you see an error message or did not enter any values, your normal distribution curve(s) follow below:")
    
    if len(means) == len(stdev):
        for i in range(len(means)):
            ax=sns.distplot(np.random.normal(means[i],stdev[i],1000), hist=False, label = i+1)
            plt.xlabel("Values")
            plt.ylabel("Probability")
            i=i+1
    else:
        print("ERROR!!: Check your entries. You did not enter same number of mean and same number of standard deviation values!")


# In[ ]:


manual_norm_dist_generator()

