#######################################################################
#      IMPORTING & STRUCTURING SECTION     #
#######################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as graph

url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
irisdf = pd.read_csv(url, names=["S_Length","S_Width","P_Length","P_Width","Species"]) # convert csv data to panda dataframe and set column names
irisdf.set_index("Species", inplace=True) # set class as the index as this is the main values that I will be filtering by

##CREATE 3 DATA SUBSETS FOR EACH IRIS 
setosa_df = irisdf[irisdf.index == "Iris-setosa"]
versicolor_df = irisdf[irisdf.index == "Iris-versicolor"]
virginica_df = irisdf[irisdf.index == "Iris-virginica"]

#######################################################################
#      INTRODUCTION     #
#######################################################################

print ("#######################################################################")
print ("                          Introduction                                 ")
print ("#######################################################################")
print ("")
print ("")
print ("Below is a sample of the first 5 rows of the Iris dataset.")
print ("Note that the data has been adjusted to add headers and set the ""Species"" as the Index column.")
print ("")
print ("")
print (irisdf.head())
print ("")
print ("")
print ("There are 4 columns of data for each Iris in the dataset recording Sepal Length, Sepal Width, Petal Length, and Petal Width in cm.")
species_list = irisdf.index.tolist() # converts index into an array
species_count = len(set(species_list)) # removes duplicates and counts 
count_per_speciesdf = irisdf[["S_Length"]].groupby(irisdf.index).size().reset_index(name="Count") # create new df summarising counts per species
count_per_speciesdf.set_index("Species", inplace=True) # change index column back to species
count_per_species = count_per_speciesdf.iloc[0]["Count"] # get value if first cell in the Count column
total_count = count_per_speciesdf["Count"].sum()
print ("")
print ("")
print (f"The primary purpose of this project is to analyse the differences between these {species_count} species.") #f allows for a variable to be insterted into string
print ("")
print ("")
print (f"As shown below, there are {count_per_species} rows of data samples for each species amounting to {total_count} rows of data in total.")
print ("")
print (count_per_speciesdf)
#if isinstance(setosa_df, pd.DataFrame): # check if something is a dataframe
  #print ("Yup")
#else: 
  #print ("Nope")
