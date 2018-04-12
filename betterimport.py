# better import setup 12-4-18

#######################################################################
#      IMPORTING & STRUCTURING SECTION     #
#######################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as graph

url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
irisdf = pd.read_csv(url, names=["S_Length","S_Width","P_Length","P_Width","I_Class"]) # convert csv data to panda dataframe and set column names
irisdf.set_index("I_Class", inplace=True) # set class as the index as this is the main values that I will be filtering by

irisdf_correlation = irisdf.corr() # create correlation table
