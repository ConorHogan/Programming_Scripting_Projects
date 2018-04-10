#######################################################################
#      IMPORTING & STRUCTURING SECTION     #
#######################################################################
import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
df = pd.read_csv(url) # apparently you call this a df for dataframe
df.columns = ["S_Length","S_Width","P_Length","P_Width","I_Class"] # assign column titles 


Sepal_Length = df["S_Length"] #create variable for use in functions
Sepal_Width = df["S_Width"] #create variable for use in functions
Petal_Length = df["P_Length"] #create variable for use in functions
Petal_Width = df["P_Width"] #create variable for use in functions
Class = df["I_Class"]

Filt_Setosa = df[(df.I_Class == 'Iris-setosa')] # data filtered for just Iris-setosa
Filt_Versicolor = df[df.I_Class == 'Iris-versicolor'] # data filtered for just Iris-versicolor
Filt_Virginica = df[df.I_Class == 'Iris-virginica'] # data filtered for just Iris-virginica
## LINK FOR FILTERS https://pythonspot.com/pandas-filter/ ###

#######################################################################
#      FUNCTIONS SECTION      #
#######################################################################

classinput = input("Enter the class you would like to filter by: ")

def filter(input("Enter the class you would like to filter by: ")):
  filtered_class = df[(df.P_Width == classinput)]
  print(filtered_class)
