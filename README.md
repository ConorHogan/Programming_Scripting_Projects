# 1.0 OVERVIEW




# 2.0 INTRODUCTION TO FISHER'S IRIS DATASET
Fisher's Iris dataset describes the biological characterisitics of three species of iris. The dataset includes 50 examples of each species, with each example having four measured characteristics:
(1) sepal length in cm
(2) sepal width in cm
(3) petal lenght in cm
(4) petal width in cm

The dataset has become a standard used to test new methods of sorting data, for example in machine learning. 
The dataset represents a difficult case for unsupervised analysis i.e. if the species for each example is not know. As I will show in my own analysis, Iris Setosa is a clearly distinct species, whereas Iris Versicolor and Iris Virginica are more difficult to distinguish from eachother for certain without knowing the species name in advance. 

### Sources for this section: 
https://www.techopedia.com/definition/32880/iris-flower-data-set
http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf
https://en.wikipedia.org/wiki/Iris_flower_data_set

# 3.0 CODE USED FOR ANALYSIS

## 3.1 INTRODUCTION TO THIS SECTION
In this section I will show the types of analysis I performed on the dataset and the Python code I used in for each type. Please note the code I have written is all contained within the one "Iris_Analysis.py" file available in the respository. The code is designed to be run in one go and it will output all the analysis outlined below to either the users terminal or as Matplotlib graphs.

## 3.2 PACKAGES USED IN ANALYSIS 
I imported the following packages for in my analysis: 

````python 
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
````
### 3.2.1 PANDAS
Panadas allows you to create spreadsheet format dataframes which can then be used to perform analysis on rows and columns in that dataframe, assign headers to columns, and also easily create subsets or pivots of that data. I previously came across Pandas when investigating how to tackle a project in work that I did at the start of this course and looked at it more closely when beginning this project. It quickly become obvious that it would be very useful for the types of analysis I would be trying to perform, especially creating sliced or summarised table to graph.
To get up to speed on how to use Pandas I watched the [Data Analysis with Python and Pandas](https://www.youtube.com/watch?v=Iqjy9UqKKuo&list=PLQVvvaa0QuDc-3szzjeP6N6b0aDrrKyL-) introductory tutorial on YouTube which was very useful. I also used the official [Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html), but this was a bit more difficult to digest.

### 3.2.2 NUMPY, MATH, & STATISTICS
NumPy and Math were imported for calculating averages, standard deviation, variance etc. I later also added the Statistics package as there is no Mode function in either NumPy or Math.

### 3.2.3 MATPLOTLIB, SEABORN, SCIPY
These three packages were used for graphing results or output of analysis. All the graphs in this project use Seaborn which builds on top of Matplotlib. The SciPy cluster package was imported for use in building the dendrograms in the cluster analysis section of this project. 
Two resources I found extremely useful when building the graphs in this project, were the [The Python Graph Gallery](http://python-graph-gallery.com) and to a lesser extent the [Seaborn official documentation](https://seaborn.pydata.org/).

## 3.3 IMPORTING THE DATASET
The source I used for the Iris dataset was the .CSV file available on the UC Irvine Machine Learning Reporistory Site (http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
The initial plan was to import my data directly from here, but the website notes that there are two mistakes in the dataset at row 35 and 38. I therefore decided to instead copy the data into a .CSV file, make the two correction outlined on the site and save the .CSV file in this repository [here](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Iris_Data.csv). 
I then used the Pandas package to read the ammended dataset directly from the repositroy into a Pandas dataframe. See code below:

````python
url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
irisdf = pd.read_csv(url, header=None) 
irisdf.columns = ["S_Length","S_Width","P_Length","P_Width","Species"]
irisdf.columns.name = "Attributes" # added this to help with stacking https://www.youtube.com/watch?v=reTeOfEebeA
irisdf.set_index("Species", inplace=True)

##CREATE 3 DATA SUBSETS FOR EACH IRIS SPECIES
setosa_df = irisdf[irisdf.index == "Iris-setosa"]
versicolor_df = irisdf[irisdf.index == "Iris-versicolor"]
virginica_df = irisdf[irisdf.index == "Iris-virginica"]
````

The above code created the "irisdf" dataframe that I would be using as master dataframe for the rest of the analysis. The code first creates a variable storing the link to the dataset file as a URL. It then creates a dataframe by reading the data in the URL. "Header=None" ensures that the top row of the dataset is not counted as header row. If I had not of done this, the top row of the dataset would have been skipped in all my future calculations. I then assigned names to each of the columns in the dataset as these were missing from the original file. I also gave the column header rows the name "Attributes" to help with pivots the data using "stack" later in my analysis. Finally, I set the "Species" column as the dataframes Index as this would be column I would be using to filter, or slice the dataframe in most of my analysis. Setting an index also removes the count column when printing the dataframe. 

I also created three seperate dataframes for each species, but in the end the were not required. 

## 3.4 BASIC ANALYSIS

### 3.4.1  PRINTING THE HEAD 
The first thing I did when examing the new dataframe was to us the "dataframe.head()" function to print the first 5 rows of the dataset and check my index and column headers had been saved correctly. 

````python
print("##########################")
print("       SAMPLE DATA        ")
print("##########################")
print ("")
print (irisdf.head())
print ("")
````
**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Dataframe_head.png)

Using the index function had moved the "Species" column for the rightmost column to the leftmost column and the column headers has been assigned correctly.

### 3.4.2 CHECKING COUNTS
To check the total count of rows in the dataset, the number of unique Species, and the count of data samples per species were correct I created the following code. 

````python
species_list = irisdf.index.tolist() # converts index into an array
species_count = len(set(species_list)) # removes duplicates and counts 
count_per_speciesdf = irisdf[["S_Length"]].groupby(irisdf.index).size().reset_index(name="Count") # create new df summarising counts per species
count_per_speciesdf.set_index("Species", inplace=True) # change index column back to species
count_per_species = count_per_speciesdf.iloc[0]["Count"] # get value if first cell in the Count column
total_count = count_per_speciesdf["Count"].sum()
print ("")
print ("")
print (f"There are {species_count} unique species.") #f allows for a variable to be insterted into string
print ("")
print ("")
print (f"As shown below, there are {count_per_species} rows of data samples for each species amounting to {total_count} rows of data in total.")
print ("")
print("##########################")
print("     ROWS PER SPECIES     ")
print("##########################")
print ("")
print (count_per_speciesdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Rowsper_species.png)

The above code first gets a count of the unique species by converting the Index into a list, then getting the list of unique values in that list using the "set()" function, and finally counting the number of these values using "len()". This is used to created the *"species_count"* variable that is inserted into the output text.

The code also creates a new dataframe ("count_per_speciesdf") summarising the count of rows for each species. The "groupby" function allows you to choose a column to summarise the data by, (in this case the Index: "Species") and perform a calculation. Here I used the "size()" function to get the count of the data in the column and then add this data to a column in the new dataframe called "Count". 

I also created two variables; the first to get first value in the "Count" column using "iloc" row plus column name which equates to the count of rows per Species, and the second to get the sum of the "Count" column which equates to the count of rows for all species. 

Finally, I insterted the variables in to the a string to print using the "f-strings" feature and also printed the "count_per_speciesdf" dataframe. 

### 3.4.3 STATISTICS FUNCTIONS
I next moved on to doing some statistical calculation on the dataset. While I count of used the groupby and calculate method I used above to create dataframes summarising the data in different way, I instead opted to create function. I did this mainly to use some of what we he learned on the course. The functions I create were used to calculate the range(max/min), median, average, standard deviation, mode, and variance for each of the four "Attribute" columns in the dataset. 

Each of the functions work in roughly the same way:

1. The user enters a dataframe to pass through the function

2. The column names are converted into a list to use a reference

3. To function iterates through each column and performs the calculation converting the result(s) and column name into a tuple(a fixed list).

4. The tuple is coverted to a list.

5. Items are taken from the list and insterted into a string which is printed. I used this tutorial (https://www.digitalocean.com/community/tutorials/how-to-use-string-formatters-in-python-3) to learn how to get the text alignment and formating of floating point numbers to include decimal places. 

#### 3.4.3.1 MIN AND MAX 
````python
print("################################")
print("     MIN AND MAX PER COLUMN     ")
print("################################")
print ("")
#CALCULATE MIN MAX OF EACH COLUMN IN DATATFRAME AND PRINT
def maxandmin(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columntuple = str(header), str(min(dataframe[header])), str(max(dataframe[header]))
    columnlist = list(columntuple)
    print("Min & Max for {0:>8}: {1:>6} {2:>6}".format(*columnlist)) # reference https://www.digitalocean.com/community/tutorials/how-to-use-string-formatters-in-python-3
maxandmin(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/MinMax.png)

#### 3.4.3.2 MEDIAN
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Median.png)

#### 3.4.3.3 MODE
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Mode.png)

#### 3.4.3.4 AVERAGE 
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Average.png)

#### 3.4.3.5 STANDARD DEVIATION
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Deviation.png)

#### 3.4.3.6 VARIANCE
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Variance.png)

### 3.5 CORROLATION ANALYSIS
After completing the above basic analysis, I then moved on to looking at if any of the attributes influenced other attributes. I did this using corrolation analysis.

#### 3.5.1 CORROLATION TABLE
There method I used to identify any potential corrolation was creating a corrolation dataframe using the "dataframe.corr()" function and then printing the dataframe.
````python
#####################
#CORROLATION TABLE
######################
corrdf = irisdf.corr() # create dataframe for corrolation
print(corrdf.round(2)) #print rounded to two decimal places to match heatmap
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrtable.png)

#### 3.5.2 CORROLATION HEATMAP
````python
#####################
#CORROLATION HEATMAP
######################
heat = graph.heatmap(corrdf, xticklabels=corrdf.columns,yticklabels=corrdf.columns, annot=True, cmap= "bwr") #https://www.youtube.com/watch?v=bA7ZcNmhnTs showed the annotate trick
plot.show(heat)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrheatmap.png)

#### 3.5.3 CORROLATION PAIRPLOT
````python
#https://python-graph-gallery.com/110-basic-correlation-matrix-with-seaborn/ & https://python-graph-gallery.com/111-custom-correlogram/
pairplotdf = irisdf.reset_index() # had to remove index to get the "hue" to pick up the seperate species. Didn't work with df.index
corr_pairplot = graph.pairplot(pairplotdf, kind="scatter", hue="Species")
plot.show(corr_pairplot)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrplot.png)

### 3.6 COMPARING / CONTRASTING SPECIES

#### 3.6.1 AVERAGES COMPARISON
````python
# CREATING AVERAGES STACKED DATAFRAME
iris_stackv = irisdf.stack() # restructure dataframe using stack
iris_stackdf = pd.DataFrame(iris_stackv, columns= ["Measures"])
iris_stackavgdf = iris_stackdf.groupby(['Species', 'Attributes']).mean().reset_index() # remove index to make axis easier to assign

#AVERAGES COMPARISON
grouped_avgs_graph = graph.factorplot(x='Species', y='Measures', hue='Attributes', data=iris_stackavgdf, kind='bar')
plot.show(grouped_avgs_graph)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/averagesbarchart.png)

#### 3.6.2 ATTRIBUTES SWARMPLOT
````python
#SWARM DISTRIBUTION
iris_stackswarm = iris_stackdf.reset_index()
swarm_by_attr_graph = graph.swarmplot(x="Attributes", y="Measures", hue="Species", data=iris_stackswarm)
plot.show(swarm_by_attr_graph)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/scatterplot.png)

### 3.7 CLUSTER ANALYSIS

#### 3.7.1 DENDROGRAM
````python
#####################
#DENDROGRAM
###################### 
# see https://python-graph-gallery.com/400-basic-dendrogram/  
del irisdf.index.name
irisdf.columns = [''] * len(irisdf.columns) #remove column names
Z = hierarchy.linkage(irisdf, 'ward') # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ explain linkage
dendrogram_chart = hierarchy.dendrogram(Z, orientation="left", labels=irisdf.index, color_threshold=10, above_threshold_color="grey")
plot.show(dendrogram_chart)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/irisdendrogram.png)

#### 3.7.2 CLUSTERMAP
````python
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
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/dendheatmap.png)
