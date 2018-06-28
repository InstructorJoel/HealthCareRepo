
# coding: utf-8

# In[1]:


##function manual_norm_dist_plotter
#function will generate up to 5 normal distribution curves given user entries of mean and standard deviation


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

