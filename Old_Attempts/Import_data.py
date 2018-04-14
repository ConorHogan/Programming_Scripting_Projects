#import dataset from git file using pandas
#reference: https://stackoverflow.com/questions/23464138/downloading-and-accessing-data-from-github-python

import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
df = pd.read_csv(url) # apparently you call this a df for dataframe
df.columns = ["S-Length","S-Width","P-Length","P-Width","Class"]

meanSL = df["S-Length"].mean()
meanSW = df["S-Width"].mean()
meanPL = df["P-Length"].mean()
meanPW = df["P-Width"].mean()

print(meanSL,meanSW,meanPL,meanPW)
