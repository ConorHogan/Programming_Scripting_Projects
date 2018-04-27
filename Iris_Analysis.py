#######################################################################
#      IMPORTING & STRUCTURING SECTION     #
#######################################################################
import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plot
import seaborn as graph
from scipy.cluster import hierarchy # for dendrogram
import statistics # for mode



url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
irisdf = pd.read_csv(url, header=None) # apparently you call this a df for dataframe
irisdf.columns = ["S_Length","S_Width","P_Length","P_Width","Species"]
irisdf.columns.name = "Attributes" # added this to help with stacking https://www.youtube.com/watch?v=reTeOfEebeA
irisdf.set_index("Species", inplace=True)

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
print("##########################")
print("       SAMPLE DATA        ")
print("##########################")
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
print("##########################")
print("     ROWS PER SPECIES     ")
print("##########################")
print ("")
print (count_per_speciesdf)
#if isinstance(count_per_speciesdf, pd.DataFrame): # check if something is a dataframe
  #print ("Yup")
#else: 
  #print ("Nope")
print ("")
print("################################")
print("     MIN AND MAX PER COLUMN     ")
print("################################")
print ("")
#CALCULATE MIN MAX OF EACH COLUMN IN DATATFRAME AND PRINT - this function took 2 hours to write :)
#Im aware summarising using a slices dataframe would be easier but I wanted a reusable function
def maxandmin(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columntuple = str(header), str(min(dataframe[header])), str(max(dataframe[header]))
    columnlist = list(columntuple)
    print("Min & Max for {0:>8}: {1:>6} {2:>6}".format(*columnlist)) # reference https://www.digitalocean.com/community/tutorials/how-to-use-string-formatters-in-python-3
maxandmin(irisdf)
print ("")
print("###############################")
print("       MEDIAN PER COLUMN       ")
print("###############################")
print ("")
def columnsmedian(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnsmedtuple = str(header), float(dataframe[header].median()) 
    columnsmedlist = list(columnsmedtuple)
    print("Median of {0:>8}: {1:>6.2f}".format(*columnsmedlist))
columnsmedian(irisdf)
print ("")
print ("")
print("################################")
print("       AVERAGE PER COLUMN       ")
print("################################")
print ("")
def columnaverage(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnavgtuple = str(header), float(dataframe[header].mean()) # mean includes decimal so must be set as float. Importing the decimal function was causing trouble
    columnavglist = list(columnavgtuple)
    print("Average(mean) of {0:>8}: {1:>6.2f}".format(*columnavglist))
columnaverage(irisdf)
print ("")
print ("")
print("###########################################")
print("       STANDARD DEVIATION PER COLUMN       ")
print("###########################################")
print ("")
def columnstddev(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnstddevtuple = str(header), float(dataframe[header].std()) 
    columnstddevlist = list(columnstddevtuple)
    print("Standard dev of {0:>8}: {1:>6.2f}".format(*columnstddevlist))
columnstddev(irisdf)
print ("")
print ("")
print("###########################################")
print("             MODE PER COLUMN               ")
print("###########################################")
print ("")
def columnsmode(dataframe):
  columnslist = dataframe.columns.values.tolist()
  for header in columnslist or []:
    columnmodetuple = str(header), str(dataframe[header].mode()) #had to make this a string to work
    columnsmodelist = list(columnmodetuple)
    print("Mode of {0:>8}: {1:>6}".format(*columnsmodelist))
columnsmode(irisdf)
print ("")
print ("")
print("#################################")
print("       VARIANCE PER COLUMN       ")
print("#################################")
print ("")
def columnsvar(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnvartuple = str(header), float(dataframe[header].var()) 
    columnsvarlist = list(columnvartuple)
    print("Variance of {0:>8}: {1:>6.2f}".format(*columnsvarlist))
columnsvar(irisdf)
####################################################################################
#                               SLICED DATAFRAMES SECTION                          #
####################################################################################
print ("")
print ("")
print ("#######################################################################")
print ("                        Sliced Dataframes                              ")
print ("#######################################################################")
print ("")
# CREATING AVERAGES STACKED DATAFRAME
iris_stackv = irisdf.stack() # restructure dataframe using stack
iris_stackdf = pd.DataFrame(iris_stackv, columns= ["Measures"])
iris_stackavgdf = iris_stackdf.groupby(['Species', 'Attributes']).mean().reset_index() # remove index to make axis easier to assign

#AVERAGES COMPARISON
grouped_avgs_graph = graph.factorplot(x='Species', y='Measures', hue='Attributes', data=iris_stackavgdf, kind='bar')
plot.show(grouped_avgs_graph)

#SWARM DISTRIBUTION
iris_stackswarm = iris_stackdf.reset_index()
swarm_by_attr_graph = graph.swarmplot(x="Attributes", y="Measures", hue="Species", data=iris_stackswarm)
plot.show(swarm_by_attr_graph)

#####################
#CORROLATION ANALYSIS
######################
print ("")
print ("")
print ("#######################################################################")
print ("                        CORROLATION ANALYSIS                           ")
print ("#######################################################################")
print ("")

#####################
#CORROLATION TABLE
######################
corrdf = irisdf.corr() # create dataframe for corrolation
print(corrdf.round(2)) #print rounded to two decimal places to match heatmap

#####################
#CORROLATION HEATMAP
######################
heat = graph.heatmap(corrdf, xticklabels=corrdf.columns,yticklabels=corrdf.columns, annot=True, cmap= "bwr") #https://www.youtube.com/watch?v=bA7ZcNmhnTs showed the annot trick
plot.show(heat)


#####################
#CORROLATION PAIRPLOT
######################
#https://python-graph-gallery.com/110-basic-correlation-matrix-with-seaborn/ & https://python-graph-gallery.com/111-custom-correlogram/
pairplotdf = irisdf.reset_index() # had to remove index to get the "hue" to pick up the seperate species. Didn't work with df.index
corr_pairplot = graph.pairplot(pairplotdf, kind="scatter", hue="Species")
plot.show(corr_pairplot)

print ("")
print ("")
print ("#######################################################################")
print ("                        CLUSTER ANALYSIS                               ")
print ("#######################################################################")
print ("")

#####################
#DENDROGRAM
###################### 
# see https://python-graph-gallery.com/400-basic-dendrogram/  
del irisdf.index.name
irisdf.columns = [''] * len(irisdf.columns) #remove column names
Z = hierarchy.linkage(irisdf, 'ward') # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ explain linkage
dendrogram_chart = hierarchy.dendrogram(Z, orientation="left", labels=irisdf.index, color_threshold=10, above_threshold_color="grey")
plot.show(dendrogram_chart)

#####################
#CLUSTERMAP
###################### 
#http://seaborn.pydata.org/generated/seaborn.clustermap.html - documentation for this one is really good, lots of examples
irisdf.index.name = "Species" # restore index name
clusterdf = irisdf.reset_index() # strip index again
species = clusterdf.pop("Species") # cuts Species column
lut = dict(zip(species.unique(), "rbg")) # assigns values to unique species
row_colors = species.map(lut) # finds species names in graph and assigns colours
clustermap = graph.clustermap(clusterdf, method="average", cmap="mako", linewidths=.5, figsize=(10, 15), row_colors=row_colors)
for tick_label in clustermap.ax_heatmap.axes.get_yticklabels(): # make colour of labels match Species row colour https://stackoverflow.com/questions/47292737/change-the-color-for-ytick-labels-in-seaborn-clustermap
    tick_text = tick_label.get_text()
    species_name = species.loc[int(tick_text)]
    tick_label.set_color(lut[species_name])
plot.show(clustermap)
